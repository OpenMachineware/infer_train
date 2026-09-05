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
# fused `matmul_quantized_cpu` - which dequantizes per block INSIDE the
# matmul kernel, so the dequantized values never leave the kernel scope
# and the weight's resident footprint stays its on-disk (Q4) size.  The
# fp16 case runs the existing threaded weight-major kernel unchanged.
#
# Layering note: this module sits between `tensor` and the CPU kernels
# (`matmul_cpu`), and is imported by the attention/transformer layers -
# never the other way around (Mojo 1.0 rejects circular imports).

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from ..cpu.matmul_cpu import (
    matmul_weight_cpu_threaded,
    matmul_quantized_cpu,
)
from .quant_types import QuantType
from std.utils.static_tuple import StaticTuple


struct QWeight(Copyable, ImplicitlyCopyable, Movable):
    """A projection weight kept in its on-disk format (Q4-resident).

    See the module docstring for the two payload variants.  `n_out` /
    `n_in` are the ELEMENT dimensions of the [out, in] matrix (the
    quantized payload's second dim is in BYTES, not elements).
    """

    var data: Tensor[DType.uint8, 2]  # quantized bytes (empty if fp16)
    var fp16: Tensor[DType.float16, 2]  # materialized (empty if quantized)
    var ggml_type: Int  # GGUF ggml type of the payload
    var quantized: Bool
    var n_out: Int
    var n_in: Int

    def __init__(out self):
        self.data = Tensor[DType.uint8, 2](StaticTuple[Int, 2](0, 0))
        self.fp16 = Tensor[DType.float16, 2](StaticTuple[Int, 2](0, 0))
        self.ggml_type = 1
        self.quantized = False
        self.n_out = 0
        self.n_in = 0

    def shape2(self) -> StaticTuple[Int, 2]:
        """The element shape [n_out, n_in] (independent of the payload)."""
        return StaticTuple[Int, 2](self.n_out, self.n_in)

    def proj(
        self,
        x: Tensor[DType.float16, 2],
        dummy_scale: Tensor[DType.float16, 1],
    ) -> Tensor[DType.float16, 2]:
        """y = W @ x with on-demand dequantization.

        Quantized weights go through the fused per-block-dequant matmul
        (`matmul_quantized_cpu`; `dummy_scale` satisfies its generic
        signature - GGUF block formats keep their scales inside the
        blocks, so the argument is ignored).  Materialized fp16 weights
        go through the threaded weight-major kernel.
        """
        if not self.quantized:
            return matmul_weight_cpu_threaded[DType.float16](x, self.fp16)
        return quant_proj_dispatch(x, self, dummy_scale)


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
) -> Tensor[DType.float16, 2]:
    """Runtime dispatch on the GGUF type -> comptime-specialized fused
    quantized matmul (per-block dequantization inside the kernel).

    The erased (registry) interface cannot carry the comptime
    `quant_type`, so each branch fixes it as a compile-time parameter -
    the same pattern as `_dequantize_dispatch` in `dequantize.mojo`.
    """
    if w.ggml_type == 12:  # Q4_K (Q4_K_M)
        return matmul_quantized_cpu[DType.float16, QuantType.Q4_K_M, 32](
            x, w.data, dummy_scale
        )
    if w.ggml_type == 2:  # Q4_0
        return matmul_quantized_cpu[DType.float16, QuantType.Q4_0, 32](
            x, w.data, dummy_scale
        )
    if w.ggml_type == 13:  # Q5_K
        return matmul_quantized_cpu[DType.float16, QuantType.Q5_K, 32](
            x, w.data, dummy_scale
        )
    if w.ggml_type == 14:  # Q6_K
        return matmul_quantized_cpu[DType.float16, QuantType.Q6_K, 32](
            x, w.data, dummy_scale
        )
    if w.ggml_type == 8:  # Q8_0
        return matmul_quantized_cpu[DType.float16, QuantType.Q8_0, 32](
            x, w.data, dummy_scale
        )
    if w.ggml_type == 23:  # IQ4_XS
        return matmul_quantized_cpu[DType.float16, QuantType.IQ4_XS, 32](
            x, w.data, dummy_scale
        )
    unimplemented(
        "qweight: unsupported quantized ggml type " + String(w.ggml_type)
    )
    return tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](0, 0))
