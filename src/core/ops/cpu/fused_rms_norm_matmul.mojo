# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/fused_rms_norm_matmul.mojo
#
# Fused RMSNorm + MatMul kernel for decode mode.
# This avoids the intermediate memory write/read of the normalized tensor.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from ...cpu_features import CpuFlags
from std.math import sqrt
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from .simd.simd_base import vec_dot_qk_q8k
from ..quantized.quant_types import QuantType, block_bytes
from .matmul_q8k_threaded import quantize_row_to_q8_k

comptime QK_K = 256


def fused_rms_norm_matmul_q8k[
    quant_type: QuantType,
](
    x: Tensor[DType.float16, 2],
    norm_weight: Tensor[DType.float16, 1],
    w_quant: Tensor[DType.uint8, 2],
    flags: CpuFlags,
    eps: Float32 = Float32(1e-5),
) -> Tensor[DType.float16, 2]:
    """Fused RMSNorm + Quantized MatMul using Q8_K activation.

    This kernel fuses RMSNorm with the subsequent matmul to avoid
    the intermediate memory write/read of the normalized tensor.

    For each row:
    1. Compute RMSNorm in-place (no intermediate buffer)
    2. Quantize normalized values to Q8_K
    3. Compute dot products with weight blocks

    Expected speedup: ~1.3-1.5x for the RMSNorm+MatMul phase.
    """
    var M = x.shape()[0]
    var K = x.shape()[1]
    var N = w_quant.shape()[0]

    var bb = block_bytes(quant_type)
    var nb = K // QK_K

    var out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, N))

    # Allocate Q8_K buffer for one row
    var q8k_buf = unsafe_alloc[UInt8](nb * 292)

    for i in range(M):
        # Step 1: Compute RMSNorm statistics
        var base = i * K
        var ss = Float32(0)

        # SIMD sum of squares
        comptime W = 8
        var d_main = (K // W) * W
        var j = 0
        while j < d_main:
            var v = x.data().unsafe_load[width=W](offset=base + j)
            var v_f32 = v.cast[DType.float32]()
            ss += (v_f32 * v_f32).reduce_add()
            j += W
        while j < K:
            var v = Float32(x.get(base + j))
            ss += v * v
            j += 1

        var rms = sqrt(ss / Float32(K) + eps)
        var inv = Float32(1) / rms

        # Step 2: Quantize normalized values to Q8_K
        # We compute normalized values on-the-fly during quantization
        for b in range(nb):
            var block_start = b * QK_K
            var block_dst = q8k_buf.unsafe_offset(b * 292)

            # Find max absolute value of normalized values
            var amax = Float32(0)
            for j in range(QK_K):
                var v = Float32(x.get(base + block_start + j))
                var v_norm = v * inv * Float32(norm_weight.get(block_start + j))
                var ax = abs(v_norm)
                if ax > amax:
                    amax = ax

            if amax == 0:
                block_dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(val=Scalar[DType.float32](0))
                continue

            # Scale to [-127, 127] range
            var iscale = 127.0 / amax
            var d = amax / 127.0

            # Store scale
            block_dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(val=Scalar[DType.float32](d))

            # Quantize normalized values and store int8
            var qs_ptr = block_dst.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
            for j in range(QK_K):
                var v = Float32(x.get(base + block_start + j))
                var v_norm = v * inv * Float32(norm_weight.get(block_start + j))
                var v_int = Int(round(iscale * v_norm))
                if v_int > 127:
                    v_int = 127
                if v_int < -127:
                    v_int = -127
                qs_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int8](v_int))

            # Compute partial sums for Q8_K
            var bsums_ptr = block_dst.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
            for j in range(16):
                var sum = Int16(0)
                for ii in range(16):
                    var qv = qs_ptr.unsafe_offset(j * 16 + ii).unsafe_load()
                    sum += Int16(qv)
                bsums_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int16](sum))

        # Step 3: Compute dot products with weight blocks
        for j in range(N):
            var sumf = Float32(0)
            for b in range(nb):
                var w_block = w_quant.data().unsafe_offset(j * nb * bb + b * bb)
                var q8_block = q8k_buf.unsafe_offset(b * 292)
                sumf += vec_dot_qk_q8k(quant_type, w_block, q8_block, flags)

            out.data().unsafe_offset(i * N + j).unsafe_store(val=Scalar[DType.float16](sumf))

    q8k_buf.unsafe_free()
    return out


def fused_rms_norm_matmul_q8k_dynamic(
    x: Tensor[DType.float16, 2],
    norm_weight: Tensor[DType.float16, 1],
    w_quant: Tensor[DType.uint8, 2],
    ggml_type: Int,
    flags: CpuFlags,
    eps: Float32 = Float32(1e-5),
) -> Tensor[DType.float16, 2]:
    """Runtime dispatch for fused RMSNorm + MatMul."""
    if ggml_type == 12:
        return fused_rms_norm_matmul_q8k[QuantType.Q4_K_M](x, norm_weight, w_quant, flags, eps)
    elif ggml_type == 13:
        return fused_rms_norm_matmul_q8k[QuantType.Q5_K](x, norm_weight, w_quant, flags, eps)
    elif ggml_type == 14:
        return fused_rms_norm_matmul_q8k[QuantType.Q6_K](x, norm_weight, w_quant, flags, eps)
    elif ggml_type == 11:
        return fused_rms_norm_matmul_q8k[QuantType.Q2_K](x, norm_weight, w_quant, flags, eps)
    elif ggml_type == 15:
        return fused_rms_norm_matmul_q8k[QuantType.Q3_K](x, norm_weight, w_quant, flags, eps)
    else:
        unimplemented("fused_rms_norm_matmul: unsupported quantization type")
        return tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](0, 0))
