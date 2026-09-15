# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/simd/q4k_q8k_dot.mojo
#
# Q4_K × Q8_K int8 dot product using NEON SDOT instruction.
#
# This is the key optimization from llama.cpp:
# - Weights stay in Q4_K (compact storage)
# - Activations are quantized to Q8_K (int8) on-the-fly
# - Hardware int8 dot product (SDOT) is used for the heavy computation
#
# On Mac M4, we have DotProd extension (FEAT_DotProd) but not i8mm (FEAT_I8MM).
# SDOT computes: sum of (int8 × int8) pairs across 128-bit vectors.

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic
from std.utils.static_tuple import StaticTuple
from ....tensor import Tensor, tensor_zeros

comptime QK_K = 256


def neon_sdot(
    acc: SIMD[DType.int32, 4],
    a: SIMD[DType.int8, 16],
    b: SIMD[DType.int8, 16],
) -> SIMD[DType.int32, 4]:
    """NEON SDOT: int8 × int8 -> int32 dot product.

    Computes: result[i] = acc[i] + sum_j(a[i*4+j] * b[i*4+j]) for i in 0..3
    Each lane processes 4 elements, so 16 elements total.
    """
    return llvm_intrinsic[
        "llvm.aarch64.neon.sdot.v4i32.v16i8",
        SIMD[DType.int32, 4],
        has_side_effect=False,
    ](acc, a, b)


def _get_scale_min_k4(
    j: Int, scales: Pointer[UInt8, MutUntrackedOrigin]
) -> Tuple[Int, Int]:
    """Unpack the 6-bit scale and min for Q4_K/Q5_K sub-block j."""
    if j < 4:
        return (
            Int(scales.unsafe_load[width=1](offset=j).value()) & 63,
            Int(scales.unsafe_load[width=1](offset=j + 4).value()) & 63,
        )
    var d = (Int(scales.unsafe_load[width=1](offset=j + 4).value()) & 0xF) | (
        (Int(scales.unsafe_load[width=1](offset=j - 4).value()) >> 6) << 4
    )
    var m = (Int(scales.unsafe_load[width=1](offset=j + 4).value()) >> 4) | (
        (Int(scales.unsafe_load[width=1](offset=j).value()) >> 6) << 4
    )
    return (d, m)


def vec_dot_q4_k_q8_k(
    # Q4_K weight block (144 bytes per 256 elements)
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    # Q8_K activation: scale at offset 0, int8 at offset 4, bsums at offset 260
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q4_K × Q8_K dot product using NEON int8 SDOT.

    Q4_K block layout (144 bytes):
    - d: fp16 scale at offset 0
    - dmin: fp16 min scale at offset 2
    - scales: 12 bytes at offset 4
    - qs: 128 bytes at offset 16 (4-bit values, 256 elements packed)

    Q8_K layout (292 bytes):
    - d: float32 scale at offset 0
    - qs: 256 int8 at offset 4
    - bsums: 16 int16 at offset 260

    Returns: dot product as float32
    """
    # Read Q4_K scales
    var w_half = w_block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(w_half.unsafe_load[width=1](offset=0).value())
    var dmin = Float32(w_half.unsafe_load[width=1](offset=1).value())
    var scales = w_block.unsafe_offset(4)
    var qs = w_block.unsafe_offset(16)

    # Read Q8_K scale
    var q8_d = q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load().value()
    var q8_qs = q8_data.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    var q8_bsums = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()

    # Compute bias: -dmin * sum(q8_bsums * min_k4)
    # This compensates for Q4_K's offset representation
    var bias = Float32(0)
    for j in range(8):
        var (_, m) = _get_scale_min_k4(j, scales)
        # Each bsum covers 16 elements, and each scale covers 32 elements
        # So for scale j, we need bsums[j*2] and bsums[j*2+1]
        var bs0 = Int32(q8_bsums.unsafe_offset(j * 2).unsafe_load().value())
        var bs1 = Int32(q8_bsums.unsafe_offset(j * 2 + 1).unsafe_load().value())
        bias -= dmin * Float32(m) * Float32(bs0 + bs1)

    # Main dot product using SDOT
    # Process 64 elements per iteration (4 scales, each covering 32 elements)
    var sumi = Int32(0)

    for j in range(8):  # 8 scales, each covers 32 elements
        var (sc, _) = _get_scale_min_k4(j, scales)

        # Load Q4_K quantized values for this scale (32 bytes = 64 elements, 4-bit each)
        # But we only need 32 elements per scale
        var q4_offset = (j // 2) * 32 + (j % 2) * 16  # Alternating low/high nibbles
        var q4_bits = qs.unsafe_offset(q4_offset)

        # Unpack 4-bit values to int8
        # For even j: low nibbles; for odd j: high nibbles
        var m4b = SIMD[DType.uint8, 16](0x0F)

        # Load 16 bytes = 32 elements (4-bit each)
        var q4_bytes = q4_bits.unsafe_load[width=16](offset=0)

        var q4_lo: SIMD[DType.int8, 16]
        var q4_hi: SIMD[DType.int8, 16]

        if j % 2 == 0:
            # Low nibbles
            q4_lo = (q4_bytes & m4b).cast[DType.int8]()
        else:
            # High nibbles
            q4_lo = (q4_bytes >> SIMD[DType.uint8, 16](4)).cast[DType.int8]()

        # Load corresponding Q8_K int8 values
        var q8_offset_val = j * 32
        var q8_vals = q8_qs.unsafe_offset(q8_offset_val).unsafe_load[width=16](offset=0)

        # SDOT: int8 × int8 -> int32
        var dot = neon_sdot(SIMD[DType.int32, 4](0), q4_lo, q8_vals)

        # Load next 16 elements
        q8_vals = q8_qs.unsafe_offset(q8_offset_val + 16).unsafe_load[width=16](offset=0)

        # Need another 16 Q4 elements - load next 16 bytes
        if j % 2 == 0:
            var q4_next = qs.unsafe_offset(q4_offset + 16).unsafe_load[width=16](offset=0)
            q4_hi = (q4_next & m4b).cast[DType.int8]()
        else:
            var q4_next = qs.unsafe_offset(q4_offset + 16).unsafe_load[width=16](offset=0)
            q4_hi = (q4_next >> SIMD[DType.uint8, 16](4)).cast[DType.int8]()

        dot = neon_sdot(dot, q4_hi, q8_vals)

        # Sum the 4 int32 lanes and apply scale
        var dot_sum = dot[0] + dot[1] + dot[2] + dot[3]
        sumi += dot_sum * Int32(sc)

    # Apply super-block scales and add bias
    return d * q8_d * Float32(sumi) + bias


def matmul_q4_k_q8_k_row(
    # Q4_K weight matrix: N rows, each row has nb blocks of 144 bytes
    w_quant: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,  # blocks per row = K / 256
    N: Int,   # number of output columns
    # Q8_K activation: 292 bytes (d + qs + bsums)
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Tensor[DType.float16, 1]:
    """Compute one row of Q4_K × Q8_K matmul.

    For each weight row, compute dot product with Q8_K activation.
    Returns a tensor of N float16 values.
    """
    var out = tensor_zeros[DType.float16, 1](StaticTuple[Int, 1](N))

    for n in range(N):
        var sumf = Float32(0)
        for b in range(nb):
            var row_offset = n * nb * 144 + b * 144
            var val = vec_dot_q4_k_q8_k(
                w_quant.unsafe_offset(row_offset),
                q8_data.unsafe_offset(b * 292),
            )
            sumf += val
        out.data().unsafe_offset(n).unsafe_store(val=Scalar[DType.float16](sumf))

    return out
