# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/fused_rms_norm_ffn.mojo
#
# Fused RMSNorm + FFN (gate+up) kernel for decode mode.
# Avoids intermediate memory write/read of the normalized tensor.

from ...tensor import Tensor, tensor_zeros
from ...utils import unimplemented
from std.math import sqrt
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.utils.static_tuple import StaticTuple
from .simd.simd_neon import vec_dot_q4_k_q8_k, vec_dot_q5_k_q8_k, vec_dot_q6_k_q8_k, vec_dot_q2_k_q8_k, vec_dot_q3_k_q8_k
from ..quantized.quant_types import QuantType, block_bytes

comptime QK_K = 256


def fused_rms_norm_gate_up_q8k(
    x: Tensor[DType.float16, 2],
    norm_weight: Tensor[DType.float16, 1],
    gate_w: Tensor[DType.uint8, 2],
    up_w: Tensor[DType.uint8, 2],
    ggml_gate: Int,
    ggml_up: Int,
    eps: Float32 = Float32(1e-5),
) -> Tuple[Tensor[DType.float16, 2], Tensor[DType.float16, 2]]:
    """Fused RMSNorm + Gate/Up projection with shared Q8_K quantization.

    This kernel fuses RMSNorm with the gate and up projections to avoid
    the intermediate memory write/read of the normalized tensor.

    For each row:
    1. Compute RMSNorm inv (sum of squares, then 1/rms)
    2. For each Q8_K block, compute normalized values on-the-fly and quantize
    3. Compute both gate and up projections using the same Q8_K buffer

    Returns: (gate_out, up_out)
    """
    var M = x.shape()[0]
    var K = x.shape()[1]
    var Ng = gate_w.shape()[0]
    var Nu = up_w.shape()[0]

    var quant_gate = _ggml_to_quant_type(ggml_gate)
    var quant_up = _ggml_to_quant_type(ggml_up)
    var bb_gate = block_bytes(quant_gate)
    var bb_up = block_bytes(quant_up)
    var nb = K // QK_K

    var gate_out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, Ng))
    var up_out = tensor_zeros[DType.float16, 2](StaticTuple[Int, 2](M, Nu))

    # Allocate Q8_K buffer for one row
    var q8k_buf = unsafe_alloc[UInt8](nb * 292)

    for i in range(M):
        var base = i * K
        var x_data = x.data()
        var norm_data = norm_weight.data()

        # Step 1: Compute RMSNorm inv using SIMD
        var ss = Float32(0)
        comptime W = 8
        var d_main = (K // W) * W
        var j = 0
        while j < d_main:
            var v = x_data.unsafe_load[width=W](offset=base + j)
            var v_f32 = v.cast[DType.float32]()
            ss += (v_f32 * v_f32).reduce_add()
            j += W
        while j < K:
            var v = Float32(x_data.unsafe_offset(base + j).unsafe_load())
            ss += v * v
            j += 1

        var rms = sqrt(ss / Float32(K) + eps)
        var inv = Float32(1) / rms

        # Step 2: For each block, compute normalized values on-the-fly and quantize
        for b in range(nb):
            var block_start = b * QK_K
            var block_dst = q8k_buf.unsafe_offset(b * 292)

            # Find max absolute value of normalized values using SIMD
            var amax = Float32(0)
            j = 0
            while j + 8 <= QK_K:
                var v = x_data.unsafe_load[width=8](offset=base + block_start + j)
                var w = norm_data.unsafe_load[width=8](offset=block_start + j)
                var v_f32 = v.cast[DType.float32]()
                var w_f32 = w.cast[DType.float32]()
                var norm_v = v_f32 * SIMD[DType.float32, 8](inv) * w_f32
                # Compute abs using max(v, -v)
                var norm_neg = SIMD[DType.float32, 8](0) - norm_v
                var norm_abs = norm_v
                for k in range(8):
                    if norm_neg[k] > norm_abs[k]:
                        norm_abs[k] = norm_neg[k]
                var block_max = norm_abs.reduce_max()
                if block_max > amax:
                    amax = block_max
                j += 8
            while j < QK_K:
                var v = Float32(x_data.unsafe_offset(base + block_start + j).unsafe_load())
                var w = Float32(norm_data.unsafe_offset(block_start + j).unsafe_load())
                var norm_v = v * inv * w
                var ax = abs(norm_v)
                if ax > amax:
                    amax = ax
                j += 1

            if amax == 0:
                block_dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(val=Scalar[DType.float32](0))
                continue

            # Scale to [-127, 127] range
            var iscale = 127.0 / amax
            var d = amax / 127.0

            # Store scale
            block_dst.unsafe_bitcast[Scalar[DType.float32]]().unsafe_store(val=Scalar[DType.float32](d))

            # Quantize normalized values on-the-fly and store int8
            var qs_ptr = block_dst.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
            j = 0
            while j + 8 <= QK_K:
                var v = x_data.unsafe_load[width=8](offset=base + block_start + j)
                var w = norm_data.unsafe_load[width=8](offset=block_start + j)
                var v_f32 = v.cast[DType.float32]()
                var w_f32 = w.cast[DType.float32]()
                var norm_v = v_f32 * SIMD[DType.float32, 8](inv) * w_f32
                var scaled = norm_v * SIMD[DType.float32, 8](iscale)
                # Round and clamp manually
                var rounded = SIMD[DType.float32, 8](0)
                for k in range(8):
                    var val = scaled[k]
                    var rounded_val = Int(round(val))
                    if rounded_val > 127:
                        rounded_val = 127
                    if rounded_val < -127:
                        rounded_val = -127
                    rounded[k] = Float32(rounded_val)
                var norm_i8 = rounded.cast[DType.int8]()
                qs_ptr.unsafe_offset(j).unsafe_store(norm_i8)
                j += 8
            while j < QK_K:
                var v = Float32(x_data.unsafe_offset(base + block_start + j).unsafe_load())
                var w = Float32(norm_data.unsafe_offset(block_start + j).unsafe_load())
                var norm_v = v * inv * w
                var v_int = Int(round(iscale * norm_v))
                if v_int > 127:
                    v_int = 127
                if v_int < -127:
                    v_int = -127
                qs_ptr.unsafe_offset(j).unsafe_store(val=Scalar[DType.int8](v_int))
                j += 1

            # Compute partial sums for Q8_K
            var bsums_ptr = block_dst.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
            for jj in range(16):
                var sum = Int16(0)
                for ii in range(16):
                    var qv = qs_ptr.unsafe_offset(jj * 16 + ii).unsafe_load()
                    sum += Int16(qv)
                bsums_ptr.unsafe_offset(jj).unsafe_store(val=Scalar[DType.int16](sum))

        # Step 3: Compute gate projection
        for j in range(Ng):
            var sumf = Float32(0)
            for b in range(nb):
                var w_block = gate_w.data().unsafe_offset(j * nb * bb_gate + b * bb_gate)
                var q8_block = q8k_buf.unsafe_offset(b * 292)
                sumf += _vec_dot_dispatch(ggml_gate, w_block, q8_block)
            gate_out.data().unsafe_offset(i * Ng + j).unsafe_store(val=Scalar[DType.float16](sumf))

        # Step 4: Compute up projection (reusing Q8_K buffer)
        for j in range(Nu):
            var sumf = Float32(0)
            for b in range(nb):
                var w_block = up_w.data().unsafe_offset(j * nb * bb_up + b * bb_up)
                var q8_block = q8k_buf.unsafe_offset(b * 292)
                sumf += _vec_dot_dispatch(ggml_up, w_block, q8_block)
            up_out.data().unsafe_offset(i * Nu + j).unsafe_store(val=Scalar[DType.float16](sumf))

    q8k_buf.unsafe_free()
    return (gate_out, up_out)


@always_inline
def _ggml_to_quant_type(ggml_type: Int) -> QuantType:
    """Convert GGML type to QuantType."""
    if ggml_type == 12:
        return QuantType.Q4_K_M
    elif ggml_type == 13:
        return QuantType.Q5_K
    elif ggml_type == 14:
        return QuantType.Q6_K
    elif ggml_type == 11:
        return QuantType.Q2_K
    elif ggml_type == 15:
        return QuantType.Q3_K
    else:
        return QuantType.Q4_K_M


@always_inline
def _vec_dot_dispatch(
    ggml_type: Int,
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_block: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Dispatch to the appropriate vec_dot kernel."""
    if ggml_type == 12:
        return vec_dot_q4_k_q8_k(w_block, q8_block)
    elif ggml_type == 13:
        return vec_dot_q5_k_q8_k(w_block, q8_block)
    elif ggml_type == 14:
        return vec_dot_q6_k_q8_k(w_block, q8_block)
    elif ggml_type == 11:
        return vec_dot_q2_k_q8_k(w_block, q8_block)
    elif ggml_type == 15:
        return vec_dot_q3_k_q8_k(w_block, q8_block)
    else:
        return vec_dot_q4_k_q8_k(w_block, q8_block)
