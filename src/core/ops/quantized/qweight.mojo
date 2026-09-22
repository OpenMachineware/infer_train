# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/quantized/qweight.mojo
#
# Q4-resident weight wrapper (M11).
#
# `QWeight` is a tagged projection-weight matrix:
#
#   * `quantized == True`  -> `data` holds the raw quantized bytes
#     [n_out, bytes_per_row] (a zero-copy `Tensor[UInt8, 2]` view over the
#     GGUF memory mapping; the format metadata lives in the tensor's
#     `quantization_info` field, set by `GGUFContext.load_tensor`);
#   * `quantized == False` -> `fp16` holds a materialized [n_out, n_in]
#     matrix (small tensors such as conv1d, F16/F32 sources, or the head
#     swapped in by the finetune API).
#
# `proj` computes y = W @ x (x [M, n_in], y [M, n_out]): the quantized
# case dispatches at RUNTIME on the GGUF type to the comptime-specialized
# fused `matmul_quantized_cpu_threaded` - which dequantizes per block
# INSIDE the matmul kernel, so the dequantized values never leave the
# kernel scope and the weight's resident footprint stays its on-disk
# (Q4) size.  M12: the N output columns are split across the C thread
# pool (per-thread private dequant scratch; single-threaded fallback
# below the parallelization threshold).  The fp16 case runs the existing
# threaded weight-major kernel unchanged.
#
# Layering note: this module sits between `tensor` and the CPU kernels
# (`matmul_cpu`), and is imported by the attention/transformer layers -
# never the other way around (Mojo 1.0 rejects circular imports).

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from ...cpu_features import CpuFlags, detect_cpu_flags
from ..cpu.matmul_cpu import (
    matmul_weight_cpu_threaded,
    matmul_quantized_cpu_threaded,
)
from ..cpu.add_cpu import add_cpu_dynamic
from ..cpu.matmul_q8k import matmul_quantized_q8k, matmul_quantized_q8k_add, matmul_quantized_q8k_nrc2
from ..cpu.matmul_q8k_threaded import matmul_quantized_q8k_threaded, matmul_quantized_q8k_worksteal
from ..gpu.matmul_gpu import matmul_weight_gpu
from ..gpu.matmul_k_quant_gpu import matmul_k_quant_gpu, matmul_k_quant_gpu_cached
from ..gpu.matmul_decode_gpu import matmul_decode_gpu, matmul_decode_gpu_cached
from ..gpu.matmul_fp16_gpu import matmul_fp16_gpu
from ..gpu.gpu_runtime import get_gpu_context, upload, gpu_available
from .quant_types import QuantType
from .dequantize_fp16 import dequantize_weights_to_fp16
from max.gpu.host import DeviceBuffer, DeviceContext
from std.utils.static_tuple import StaticTuple
from std.collections.optional import Optional


struct QWeight(Copyable, ImplicitlyCopyable, Movable):
    """A projection weight kept in its on-disk format (Q4-resident).

    See the module docstring for the two payload variants.  `n_out` /
    `n_in` are the ELEMENT dimensions of the [out, in] matrix (the
    quantized payload's second dim is in BYTES, not elements).

    GPU buffer caching: quantized weights can be uploaded to GPU once
    and reused across multiple matmul calls, avoiding per-call upload
    overhead (~32MB for 7B models).
    """

    var data: Tensor[DType.uint8, 2]  # quantized bytes (empty if fp16)
    var fp16: Tensor[DType.float16, 2]  # materialized (empty if quantized)
    var ggml_type: Int  # GGUF ggml type of the payload
    var quantized: Bool
    var n_out: Int
    var n_in: Int
    var gpu_buf: Optional[DeviceBuffer[DType.uint8]]  # cached GPU buffer for quantized weights
    var fp16_dequant: Tensor[DType.float16, 2]  # dequantized FP16 weights (for GPU decode)
    var gpu_buf_fp16: Optional[DeviceBuffer[DType.float16]]  # cached GPU buffer for FP16 weights
    var fp32_dequant: Tensor[DType.float32, 2]  # dequantized FP32 weights (for batch prefill)

    def __init__(out self):
        self.data = Tensor[DType.uint8, 2](StaticTuple[Int, 2](0, 0))
        self.fp16 = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.ggml_type = 1
        self.quantized = False
        self.n_out = 0
        self.n_in = 0
        self.gpu_buf = None
        self.fp16_dequant = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.gpu_buf_fp16 = None
        self.fp32_dequant = Tensor[DType.float32, 2](StaticTuple[Int, 2](0, 0))

    def __copyinit__(out self, existing: QWeight):
        self.data = existing.data
        self.fp16 = existing.fp16
        self.ggml_type = existing.ggml_type
        self.quantized = existing.quantized
        self.n_out = existing.n_out
        self.n_in = existing.n_in
        # GPU buffers are NOT copied - each instance has its own buffers
        # This is correct because the original owner should manage the buffers
        self.gpu_buf = None
        self.fp16_dequant = existing.fp16_dequant
        self.gpu_buf_fp16 = None
        self.fp32_dequant = existing.fp32_dequant

    def shape2(self) -> StaticTuple[Int, 2]:
        """The element shape [n_out, n_in] (independent of the payload)."""
        return StaticTuple[Int, 2](self.n_out, self.n_in)

    def ensure_gpu_buf_fp16(mut self, ctx: DeviceContext) raises -> DeviceBuffer[DType.float16]:
        """Get or create GPU buffer for FP16 weights.

        If weights are quantized, dequantizes to fp16_dequant first.
        Then uploads to GPU if not already cached.

        Args:
            ctx: GPU device context

        Returns:
            GPU buffer containing FP16 weights [n_out, n_in]
        """
        # If not quantized, use fp16 directly
        if not self.quantized:
            if not self.gpu_buf_fp16:
                self.gpu_buf_fp16 = upload[DType.float16, 2](ctx, self.fp16)
            return self.gpu_buf_fp16.value()

        # If quantized, need to dequantize first
        if self.fp16_dequant.shape() != StaticTuple[Int, 2](self.n_out, self.n_in):
            # Dequantize quantized weights to FP16
            from .dequantize_fp16 import dequantize_weights_to_fp16
            self.fp16_dequant = dequantize_weights_to_fp16(self.data, self.ggml_type, self.n_out, self.n_in)

        # Upload to GPU if not already cached
        if not self.gpu_buf_fp16:
            self.gpu_buf_fp16 = upload[DType.float16, 2](ctx, self.fp16_dequant)

        return self.gpu_buf_fp16.value()

    def upload_to_gpu(mut self, ctx: DeviceContext) raises:
        """Upload quantized weights to GPU for persistent caching.

        Call this once during model initialization to avoid per-call
        upload overhead. Only applies to quantized weights.

        Args:
            ctx: DeviceContext to use for upload (should be cached at model level)
        """
        if not self.quantized or not gpu_available[DType.float16]():
            return
        self.gpu_buf = upload[DType.uint8, 2](ctx, self.data)

    def prepare_fp16_for_gpu(mut self, ctx: DeviceContext) raises:
        """Dequantize weights to FP16 and upload to GPU for fast decode.

        This is a one-time cost during model initialization. After this,
        decode mode can use FP16 GPU matmul which is much faster than
        the quantized decode kernel.

        Args:
            ctx: DeviceContext to use for upload (should be cached at model level)
        """
        if not self.quantized or not gpu_available[DType.float16]():
            return

        # Dequantize to FP16 if not already done
        if self.fp16_dequant.shape()[0] == 0:
            var quant_type = QuantType.Q4_K_M
            if self.ggml_type == 13:
                quant_type = QuantType.Q5_K
            elif self.ggml_type == 14:
                quant_type = QuantType.Q6_K
            elif self.ggml_type == 11:
                quant_type = QuantType.Q2_K
            elif self.ggml_type == 15:
                quant_type = QuantType.Q3_K

            self.fp16_dequant = dequantize_weights_to_fp16(self.data, quant_type)

        # Upload FP16 weights to GPU
        self.gpu_buf_fp16 = upload[DType.float16, 2](ctx, self.fp16_dequant)

    def dequantize_to_fp16(mut self):
        """Dequantize weights to FP16 for fast batch prefill.

        This is a one-time cost during model initialization. After this,
        batch prefill (M > 1) can use BLAS with pre-dequantized weights,
        avoiding the on-the-fly dequantization overhead (140-1550 us -> ~10 us).
        """
        if not self.quantized:
            return

        # Dequantize to FP16 if not already done
        if self.fp16_dequant.shape()[0] == 0:
            var quant_type = QuantType.Q4_K_M
            if self.ggml_type == 13:
                quant_type = QuantType.Q5_K
            elif self.ggml_type == 14:
                quant_type = QuantType.Q6_K
            elif self.ggml_type == 11:
                quant_type = QuantType.Q2_K
            elif self.ggml_type == 15:
                quant_type = QuantType.Q3_K

            self.fp16_dequant = dequantize_weights_to_fp16(self.data, quant_type)

    def dequantize_to_fp32(mut self):
        """Dequantize weights to FP32 for fast batch prefill with BLAS.

        This is a one-time cost during model initialization. After this,
        batch prefill (M > 1) can use BLAS directly without FP16→FP32 conversion.
        """
        from .dequantize import dequantize_into_f32
        from .quant_types import block_bytes, ggml_type

        if not self.quantized:
            return

        # Dequantize to FP32 if not already done
        if self.fp32_dequant.shape()[0] == 0:

            var N = self.n_out
            var K = self.n_in

            # Determine quantization type
            var quant_type = QuantType.Q4_K_M
            if self.ggml_type == 13:
                quant_type = QuantType.Q5_K
            elif self.ggml_type == 14:
                quant_type = QuantType.Q6_K
            elif self.ggml_type == 11:
                quant_type = QuantType.Q2_K
            elif self.ggml_type == 15:
                quant_type = QuantType.Q3_K

            # Dequantize directly to FP32
            self.fp32_dequant = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](N, K))

            # Use the block dequantization
            var nb = K // 256  # Number of 256-element blocks per row
            var bb = block_bytes(quant_type)
            var gtype = ggml_type(quant_type)

            # Dequantize each row
            for row in range(N):
                var row_view = Tensor[DType.float32, 2](
                    StaticTuple[Int, 2](1, K),
                    self.fp32_dequant.data().unsafe_offset(row * K).unsafe_bitcast[Scalar[DType.float32]](),
                    self.fp32_dequant.device(),
                )
                dequantize_into_f32(
                    gtype,
                    self.data.data().unsafe_offset(row * nb * bb),
                    0,
                    row_view,
                    K,
                )

    def proj(
        self,
        x: Tensor[DType.float16, 2],
        dummy_scale: Tensor[DType.float16, 1],
        use_gpu: Bool = True,  # GPU path enabled with context caching
        gpu_ctx: Optional[DeviceContext] = None,
    ) -> Tensor[DType.float16, 2]:
        """y = W @ x with on-demand dequantization.

        Quantized weights go through the fused per-block-dequant matmul
        (`matmul_quantized_cpu`; `dummy_scale` satisfies its generic
        signature - GGUF block formats keep their scales inside the
        blocks, so the argument is ignored).  Materialized fp16 weights
        go through the GPU weight-major kernel when `use_gpu` is True,
        otherwise the threaded CPU kernel.

        Args:
            x: Input tensor [M, n_in]
            dummy_scale: Placeholder for generic signature
            use_gpu: Whether to use GPU path
            gpu_ctx: Optional cached DeviceContext for GPU operations
        """
        if not self.quantized:
            if use_gpu:
                return matmul_weight_gpu[DType.float16](x, self.fp16)
            return matmul_weight_cpu_threaded[DType.float16](x, self.fp16)
        return quant_proj_dispatch(x, self, dummy_scale, use_gpu, gpu_ctx)

    def proj_fp32(
        self,
        x: Tensor[DType.float32, 2],
        use_gpu: Bool = False,
    ) -> Tensor[DType.float32, 2]:
        """FP32 matmul for batch prefill.

        Returns FP32 result directly, avoiding FP16↔FP32 conversion overhead.
        For batch prefill (M > 1), uses pre-dequantized FP32 weights + BLAS.

        Args:
            x: FP32 input tensor [M, n_in]
            use_gpu: Whether to use GPU path (currently not supported)

        Returns:
            FP32 result tensor [M, n_out]
        """
        from ..cpu.blas_cpu import matmul_weight_blas_f32

        # Non-quantized weights: convert FP16 to FP32 and use BLAS
        if not self.quantized:
            var w_f32 = tensor_zeros[DType.float32, 2](
                StaticTuple[Int, 2](self.n_out, self.n_in)
            )
            for i in range(self.n_out * self.n_in):
                w_f32.data().unsafe_offset(i).unsafe_store(
                    val=Float32(self.fp16.data().unsafe_offset(i).unsafe_load())
                )
            return matmul_weight_blas_f32(x, w_f32)

        # Batch prefill with pre-dequantized FP32 weights
        if x.shape()[0] > 1 and self.fp32_dequant.shape()[0] > 0:
            return matmul_weight_blas_f32(x, self.fp32_dequant)

        # Fallback: dequantize on-the-fly (should not happen in batch prefill)
        # Create a temporary FP32 weight tensor
        var w_f32 = tensor_zeros[DType.float32, 2](
            StaticTuple[Int, 2](self.n_out, self.n_in)
        )
        from .dequantize import dequantize_into_f32
        from .quant_types import block_bytes, ggml_type

        var K = self.n_in
        var quant_type = QuantType.Q4_K_M
        if self.ggml_type == 13:
            quant_type = QuantType.Q5_K
        elif self.ggml_type == 14:
            quant_type = QuantType.Q6_K

        var nb = K // 256
        var bb = block_bytes(quant_type)
        var gtype = ggml_type(quant_type)

        for row in range(self.n_out):
            var row_view = Tensor[DType.float32, 2](
                StaticTuple[Int, 2](1, K),
                w_f32.data().unsafe_offset(row * K).unsafe_bitcast[Scalar[DType.float32]](),
                w_f32.device(),
            )
            dequantize_into_f32(
                gtype,
                self.data.data().unsafe_offset(row * nb * bb),
                0,
                row_view,
                K,
            )

        return matmul_weight_blas_f32(x, w_f32)

    def proj_add(
        self,
        x: Tensor[DType.float16, 2],
        dummy_scale: Tensor[DType.float16, 1],
        residual: Tensor[DType.float16, 2],
        use_gpu: Bool = False,  # CPU path for now
        gpu_ctx: Optional[DeviceContext] = None,
    ) -> Tensor[DType.float16, 2]:
        """y = W @ x + residual (fused operation).

        Fused matmul + residual add saves one memory read/write pass.
        For K-quant weights, uses the fused Q8_K kernel.

        Args:
            x: Input tensor [M, n_in]
            dummy_scale: Placeholder for generic signature
            residual: Residual tensor [M, n_out]
            use_gpu: Whether to use GPU path
            gpu_ctx: Optional cached DeviceContext for GPU operations
        """
        if not self.quantized:
            # Non-quantized: fall back to separate matmul + add
            var result = matmul_weight_cpu_threaded[DType.float16](x, self.fp16)
            return add_cpu_dynamic[DType.float16](residual, result)
        return quant_proj_add_dispatch(x, self, dummy_scale, residual, use_gpu, gpu_ctx)


def qweight_from_fp16(w: Tensor[DType.float16, 2]) -> QWeight:
    """Wrap a materialized fp16 matrix as a `QWeight` (zero-copy view)."""
    var q = QWeight()
    q.fp16 = w
    q.quantized = False
    q.ggml_type = 1  # F16
    q.n_out = w.shape()[0]
    q.n_in = w.shape()[1]
    return q


def quant_proj_dispatch(
    x: Tensor[DType.float16, 2],
    w: QWeight,
    dummy_scale: Tensor[DType.float16, 1],
    use_gpu: Bool = True,
    gpu_ctx: Optional[DeviceContext] = None,
) -> Tensor[DType.float16, 2]:
    """Runtime dispatch on the GGUF type -> comptime-specialized fused
    quantized matmul (per-block dequantization inside the kernel).

    M12: the threaded variant - the N output columns are split across
    the C thread pool (per-thread private dequant scratch); below the
    parallelization threshold it falls back to the single-threaded
    kernel, so the result is bit-identical either way.

    The erased (registry) interface cannot carry the comptime
    `quant_type`, so each branch fixes it as a compile-time parameter -
    the same pattern as `_dequantize_dispatch` in `dequantize.mojo`.

    M13: for large weight matrices, use tiled BLAS (dequantize in tiles
    and use Accelerate BLAS for the heavy matmul work). This is faster
    than block-by-block SIMD for large matrices (7B+ models).

    M14: GPU quantized matmul with on-device dequantization.
    Uses cached GPU buffer and context for persistent weight storage.

    M15: Decode-optimized GPU kernel for M <= 4 with warp shuffle reduction.
    """
    from ..cpu.blas_cpu import matmul_quantized_blas_tiled, matmul_weight_blas_f32

    var flags = detect_cpu_flags()
    var M = x.shape()[0]
    var n_blocks = w.n_in // 256  # QK_K = 256

    # Fast path for batch prefill: use pre-dequantized FP32 weights with BLAS
    # This avoids the on-the-fly dequantization overhead (140-1550 us -> ~10 us)
    if M > 1 and w.fp32_dequant.shape()[0] > 0:
        # SIMD-accelerated FP16→FP32 conversion
        var K = x.shape()[1]
        var N = w.n_out
        var x_f32 = tensor_zeros[DType.float32, 2](StaticTuple[Int, 2](M, K))
        var total = M * K
        var i = 0
        while i + 16 <= total:
            var vec = x.data().unsafe_load[width=16](offset=i)
            var vec_f32 = vec.cast[DType.float32]()
            x_f32.data().unsafe_offset(i).unsafe_store(val=vec_f32)
            i += 16
        while i < total:
            x_f32.data().unsafe_offset(i).unsafe_store(
                val=Float32(x.data().unsafe_offset(i).unsafe_load())
            )
            i += 1

        # BLAS matmul with pre-dequantized FP32 weights
        var result_f32 = matmul_weight_blas_f32(x_f32, w.fp32_dequant)

        # SIMD-accelerated FP32→FP16 conversion
        var result = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))
        total = M * N
        i = 0
        while i + 16 <= total:
            var vec = result_f32.data().unsafe_load[width=16](offset=i)
            var vec_f16 = vec.cast[DType.float16]()
            result.data().unsafe_offset(i).unsafe_store(val=vec_f16)
            i += 16
        while i < total:
            result.data().unsafe_offset(i).unsafe_store(
                val=Scalar[DType.float16](result_f32.data().unsafe_offset(i).unsafe_load())
            )
            i += 1
        return result

    # Dynamic dispatch for decode mode (M <= 4)
    # Use FP16 GPU path if weights are prepared, otherwise fall back to CPU
    if use_gpu and M <= 4 and w.gpu_buf_fp16:
        # FP16 GPU path is fast (prepared weights)
        return matmul_fp16_gpu(x, w.fp16_dequant, w.gpu_buf_fp16, gpu_ctx)
    elif use_gpu and M <= 4:
        # FP16 not prepared, fall back to CPU (faster than quantized GPU kernel)
        pass
    elif use_gpu:  # Batch mode (M > 4): GPU path with on-device dequantization
        # Q4_K (ggml_type 12)
        if w.ggml_type == 12:
            return matmul_k_quant_gpu_cached[QuantType.Q4_K_M](
                x, w.data, n_blocks, w.gpu_buf, gpu_ctx
            )

        # Q5_K (ggml_type 13)
        if w.ggml_type == 13:
            return matmul_k_quant_gpu_cached[QuantType.Q5_K](
                x, w.data, n_blocks, w.gpu_buf, gpu_ctx
            )

        # Q6_K (ggml_type 14)
        if w.ggml_type == 14:
            return matmul_k_quant_gpu_cached[QuantType.Q6_K](
                x, w.data, n_blocks, w.gpu_buf, gpu_ctx
            )

        # Q2_K (ggml_type 11)
        if w.ggml_type == 11:
            return matmul_k_quant_gpu_cached[QuantType.Q2_K](
                x, w.data, n_blocks, w.gpu_buf, gpu_ctx
            )

        # Q3_K (ggml_type 15)
        if w.ggml_type == 15:
            return matmul_k_quant_gpu_cached[QuantType.Q3_K](
                x, w.data, n_blocks, w.gpu_buf, gpu_ctx
            )

    # K-quant formats: Use optimized quantized kernel for all batch sizes
    # BLAS is inefficient for small batches (M < 64) - efficiency drops to 17% at M=32
    # Quantized kernel is faster and uses less memory

    # Q4_K (ggml_type 12)
    if w.ggml_type == 12:
        # Use threaded kernel for batch prefill (faster than BLAS for M < 64)
        if x.shape()[0] > 1:
            return matmul_quantized_q8k_threaded[QuantType.Q4_K_M](x, w.data, dummy_scale, flags)
        if w.n_out >= 2048:
            return matmul_quantized_q8k_threaded[QuantType.Q4_K_M](x, w.data, dummy_scale, flags)
        # Use nrc2 optimization for Q4_K decode
        return matmul_quantized_q8k_nrc2[QuantType.Q4_K_M](x, w.data, dummy_scale, flags)

    # Q5_K (ggml_type 13)
    if w.ggml_type == 13:
        # Use threaded kernel for batch prefill
        if x.shape()[0] > 1:
            return matmul_quantized_q8k_threaded[QuantType.Q5_K](x, w.data, dummy_scale, flags)
        if w.n_out >= 2048:
            return matmul_quantized_q8k_threaded[QuantType.Q5_K](x, w.data, dummy_scale, flags)
        return matmul_quantized_q8k[QuantType.Q5_K](x, w.data, dummy_scale, flags)

    # Q6_K (ggml_type 14)
    if w.ggml_type == 14:
        # Use threaded kernel for batch prefill
        if x.shape()[0] > 1:
            return matmul_quantized_q8k_threaded[QuantType.Q6_K](x, w.data, dummy_scale, flags)
        if w.n_out >= 2048:
            return matmul_quantized_q8k_threaded[QuantType.Q6_K](x, w.data, dummy_scale, flags)
        return matmul_quantized_q8k[QuantType.Q6_K](x, w.data, dummy_scale, flags)

    # Q2_K (ggml_type 11)
    if w.ggml_type == 11:
        # Use threaded kernel for batch prefill
        if x.shape()[0] > 1:
            return matmul_quantized_q8k_threaded[QuantType.Q2_K](x, w.data, dummy_scale, flags)
        if w.n_out >= 2048:
            return matmul_quantized_q8k_threaded[QuantType.Q2_K](x, w.data, dummy_scale, flags)
        return matmul_quantized_q8k[QuantType.Q2_K](x, w.data, dummy_scale, flags)

    # Q3_K (ggml_type 15)
    if w.ggml_type == 15:
        # Use threaded kernel for batch prefill
        if x.shape()[0] > 1:
            return matmul_quantized_q8k_threaded[QuantType.Q3_K](x, w.data, dummy_scale, flags)
        if w.n_out >= 2048:
            return matmul_quantized_q8k_threaded[QuantType.Q3_K](x, w.data, dummy_scale, flags)
        return matmul_quantized_q8k[QuantType.Q3_K](x, w.data, dummy_scale, flags)

    # Non-K-quant formats: Use standard dequantize + matmul path
    if w.ggml_type == 2:  # Q4_0
        # Use threaded kernel for batch prefill
        if x.shape()[0] > 1:
            return matmul_quantized_cpu_threaded[DType.float16, QuantType.Q4_0, 32](
                x, w.data, dummy_scale
            )
        return matmul_quantized_cpu_threaded[DType.float16, QuantType.Q4_0, 32](
            x, w.data, dummy_scale
        )
    if w.ggml_type == 8:  # Q8_0
        # Use threaded kernel for batch prefill
        if x.shape()[0] > 1:
            return matmul_quantized_cpu_threaded[DType.float16, QuantType.Q8_0, 32](
                x, w.data, dummy_scale
            )
        return matmul_quantized_cpu_threaded[DType.float16, QuantType.Q8_0, 32](
            x, w.data, dummy_scale
        )
    if w.ggml_type == 23:  # IQ4_XS
        return matmul_quantized_cpu_threaded[DType.float16, QuantType.IQ4_XS, 32](
            x, w.data, dummy_scale
        )
    unimplemented(
        "qweight: unsupported quantized ggml type " + String(w.ggml_type)
    )
    return tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](0, 0))


def quant_proj_add_dispatch(
    x: Tensor[DType.float16, 2],
    w: QWeight,
    dummy_scale: Tensor[DType.float16, 1],
    residual: Tensor[DType.float16, 2],
    use_gpu: Bool = False,
    gpu_ctx: Optional[DeviceContext] = None,
) -> Tensor[DType.float16, 2]:
    """Fused matmul + residual add for K-quant weights.

    Uses the fused Q8_K kernel which computes (x @ W) + residual in one pass,
    saving one memory read/write over the output tensor.
    """
    from ..cpu.blas_cpu import matmul_quantized_blas_tiled

    var flags = detect_cpu_flags()
    var M = x.shape()[0]
    var n_blocks = w.n_in // 256

    # CPU path for decode mode (M <= 4)
    if use_gpu and M <= 4:
        pass
    elif use_gpu:
        # GPU path for batch mode - use separate matmul + add for now
        if w.ggml_type == 12:
            var result = matmul_k_quant_gpu_cached[QuantType.Q4_K_M](
                x, w.data, n_blocks, w.gpu_buf, gpu_ctx
            )
            return add_cpu_dynamic[DType.float16](residual, result)
        elif w.ggml_type == 13:
            var result = matmul_k_quant_gpu_cached[QuantType.Q5_K](
                x, w.data, n_blocks, w.gpu_buf, gpu_ctx
            )
            return add_cpu_dynamic[DType.float16](residual, result)
        elif w.ggml_type == 14:
            var result = matmul_k_quant_gpu_cached[QuantType.Q6_K](
                x, w.data, n_blocks, w.gpu_buf, gpu_ctx
            )
            return add_cpu_dynamic[DType.float16](residual, result)
        elif w.ggml_type == 11:
            var result = matmul_k_quant_gpu_cached[QuantType.Q2_K](
                x, w.data, n_blocks, w.gpu_buf, gpu_ctx
            )
            return add_cpu_dynamic[DType.float16](residual, result)
        elif w.ggml_type == 15:
            var result = matmul_k_quant_gpu_cached[QuantType.Q3_K](
                x, w.data, n_blocks, w.gpu_buf, gpu_ctx
            )
            return add_cpu_dynamic[DType.float16](residual, result)

    # K-quant formats: Use threaded matmul + separate add
    # The "fused" kernel saves one memory pass but loses threading (10x slowdown)
    # Threading gives much bigger speedup than the fusion optimization
    if w.ggml_type >= 11 and w.ggml_type <= 15:
        var result = quant_proj_dispatch(x, w, dummy_scale, use_gpu, gpu_ctx)
        return add_cpu_dynamic[DType.float16](residual, result)

    # Non-K-quant formats: fallback to separate matmul + add
    var result = quant_proj_dispatch(x, w, dummy_scale, use_gpu, gpu_ctx)
    return add_cpu_dynamic[DType.float16](residual, result)
