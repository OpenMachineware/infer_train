# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/simd/simd_neon.mojo
#
# ARM NEON optimized quantized dot product kernels.
#
# These kernels compute dot products directly on quantized data,
# avoiding intermediate storage. The implementation follows llama.cpp's
# ggml_vec_dot_q series in ggml-cpu.c.

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.memory.alloc import unsafe_alloc
from std.sys import llvm_intrinsic

# NEON SIMD width for Float32
comptime NEON_WIDTH = 8  # 8x Float32 = 256 bits


def _get_scale_min_k4(
    j: Int, scales: Pointer[UInt8, MutUntrackedOrigin]
) -> Tuple[Int, Int]:
    """Unpack the 6-bit scale and min for Q4_K/Q5_K sub-block j."""
    if j < 4:
        return (
            Int(scales.unsafe_load[width=1](offset=j)) & 63,
            Int(scales.unsafe_load[width=1](offset=j + 4)) & 63,
        )
    var d = (Int(scales.unsafe_load[width=1](offset=j + 4)) & 0xF) | (
        (Int(scales.unsafe_load[width=1](offset=j - 4)) >> 6) << 4
    )
    var m = (Int(scales.unsafe_load[width=1](offset=j + 4)) >> 4) | (
        (Int(scales.unsafe_load[width=1](offset=j)) >> 6) << 4
    )
    return (d, m)


def vec_dot_q4_k_neon[
    dtype: DType
](
    x: Pointer[Scalar[dtype], MutUntrackedOrigin],
    block: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """NEON-optimized Q4_K dot product with FMA and loop unrolling.

    Q4_K block layout: d(2) dmin(2) scales(12) qs(128) = 144 bytes per 256 elements.
    Uses 4 accumulators and FMA instructions for maximum throughput.
    """
    # 4 accumulators to hide FMA latency
    var acc0 = SIMD[DType.float32, NEON_WIDTH](0)
    var acc1 = SIMD[DType.float32, NEON_WIDTH](0)
    var acc2 = SIMD[DType.float32, NEON_WIDTH](0)
    var acc3 = SIMD[DType.float32, NEON_WIDTH](0)
    var k = 0

    # Process 4 blocks at a time for better instruction-level parallelism
    var blk_idx = 0
    while blk_idx + 3 < nb:
        for blk_off in range(4):
            var blk = blk_idx + blk_off
            var blk_half = block.unsafe_offset(blk * 144).unsafe_bitcast[Scalar[DType.float16]]()
            var blk_d = Float32(blk_half.unsafe_load[width=1](offset=0))
            var blk_dmin = Float32(blk_half.unsafe_load[width=1](offset=1))
            var blk_scales = block.unsafe_offset(blk * 144 + 4)
            var blk_qs = block.unsafe_offset(blk * 144 + 16)

            var q = blk_qs
            var acc_local = SIMD[DType.float32, NEON_WIDTH](0)

            # Fully unroll pair loop (4 pairs per block)
            var pair = 0
            while pair < 4:
                var (sc0, m0) = _get_scale_min_k4(pair * 2, blk_scales)
                var d0 = blk_d * Float32(sc0)
                var m0v = blk_dmin * Float32(m0)
                var (sc1, m1) = _get_scale_min_k4(pair * 2 + 1, blk_scales)
                var d1 = blk_d * Float32(sc1)
                var m1v = blk_dmin * Float32(m1)

                # Load 32 bytes (4 SIMD vectors), split into 64 elements
                var b0 = q.unsafe_load[width=NEON_WIDTH](offset=0)
                var b1 = q.unsafe_load[width=NEON_WIDTH](offset=NEON_WIDTH)
                var b2 = q.unsafe_load[width=NEON_WIDTH](offset=16)
                var b3 = q.unsafe_load[width=NEON_WIDTH](offset=24)

                # Low nibbles - use FMA (elements 0-15, 16-31)
                var lo0 = (b0 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
                var lo1 = (b1 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
                var lo2 = (b2 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
                var lo3 = (b3 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
                var wv_lo0 = d0 * lo0 - m0v
                var wv_lo1 = d0 * lo1 - m0v
                var wv_lo2 = d0 * lo2 - m0v
                var wv_lo3 = d0 * lo3 - m0v
                var x0 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64).cast[DType.float32]()
                var x1 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 8).cast[DType.float32]()
                var x4 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 16).cast[DType.float32]()
                var x5 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 24).cast[DType.float32]()
                acc_local = x0.fma(wv_lo0, acc_local)
                acc_local = x1.fma(wv_lo1, acc_local)
                acc_local = x4.fma(wv_lo2, acc_local)
                acc_local = x5.fma(wv_lo3, acc_local)

                # High nibbles - use FMA (elements 32-47, 48-63)
                var hi0 = (b0 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
                var hi1 = (b1 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
                var hi2 = (b2 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
                var hi3 = (b3 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
                var wv_hi0 = d1 * hi0 - m1v
                var wv_hi1 = d1 * hi1 - m1v
                var wv_hi2 = d1 * hi2 - m1v
                var wv_hi3 = d1 * hi3 - m1v
                var x2 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 32).cast[DType.float32]()
                var x3 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 40).cast[DType.float32]()
                var x6 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 48).cast[DType.float32]()
                var x7 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 56).cast[DType.float32]()
                acc_local = x2.fma(wv_hi0, acc_local)
                acc_local = x3.fma(wv_hi1, acc_local)
                acc_local = x6.fma(wv_hi2, acc_local)
                acc_local = x7.fma(wv_hi3, acc_local)

                pair += 1
                q = q.unsafe_offset(32)

            # Distribute across accumulators
            if blk_off == 0:
                acc0 = acc0 + acc_local
            elif blk_off == 1:
                acc1 = acc1 + acc_local
            elif blk_off == 2:
                acc2 = acc2 + acc_local
            else:
                acc3 = acc3 + acc_local

        k += 1024  # 4 blocks * 256 elements
        blk_idx += 4

    # Handle remaining blocks with same pattern
    while blk_idx < nb:
        var blk_half = block.unsafe_offset(blk_idx * 144).unsafe_bitcast[Scalar[DType.float16]]()
        var blk_d = Float32(blk_half.unsafe_load[width=1](offset=0))
        var blk_dmin = Float32(blk_half.unsafe_load[width=1](offset=1))
        var blk_scales = block.unsafe_offset(blk_idx * 144 + 4)
        var blk_qs = block.unsafe_offset(blk_idx * 144 + 16)

        var q = blk_qs
        for pair in range(4):
            var (sc0, m0) = _get_scale_min_k4(pair * 2, blk_scales)
            var d0 = blk_d * Float32(sc0)
            var m0v = blk_dmin * Float32(m0)
            var (sc1, m1) = _get_scale_min_k4(pair * 2 + 1, blk_scales)
            var d1 = blk_d * Float32(sc1)
            var m1v = blk_dmin * Float32(m1)

            var b0 = q.unsafe_load[width=NEON_WIDTH](offset=0)
            var b1 = q.unsafe_load[width=NEON_WIDTH](offset=NEON_WIDTH)
            var b2 = q.unsafe_load[width=NEON_WIDTH](offset=16)
            var b3 = q.unsafe_load[width=NEON_WIDTH](offset=24)

            var lo0 = (b0 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
            var lo1 = (b1 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
            var lo2 = (b2 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
            var lo3 = (b3 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
            var x0 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64).cast[DType.float32]()
            var x1 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 8).cast[DType.float32]()
            var x4 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 16).cast[DType.float32]()
            var x5 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 24).cast[DType.float32]()
            acc0 = x0.fma(d0 * lo0 - m0v, acc0)
            acc0 = x1.fma(d0 * lo1 - m0v, acc0)
            acc0 = x4.fma(d0 * lo2 - m0v, acc0)
            acc0 = x5.fma(d0 * lo3 - m0v, acc0)

            var hi0 = (b0 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
            var hi1 = (b1 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
            var hi2 = (b2 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
            var hi3 = (b3 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
            var x2 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 32).cast[DType.float32]()
            var x3 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 40).cast[DType.float32]()
            var x6 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 48).cast[DType.float32]()
            var x7 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 56).cast[DType.float32]()
            acc0 = x2.fma(d1 * hi0 - m1v, acc0)
            acc0 = x3.fma(d1 * hi1 - m1v, acc0)
            acc0 = x6.fma(d1 * hi2 - m1v, acc0)
            acc0 = x7.fma(d1 * hi3 - m1v, acc0)

            q = q.unsafe_offset(32)

        k += 256
        blk_idx += 1

    return (acc0 + acc1 + acc2 + acc3).reduce_add()


def vec_dot_q4_0_neon[
    dtype: DType
](
    x: Pointer[Scalar[dtype], MutUntrackedOrigin],
    block: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """NEON-optimized Q4_0 dot product with FMA and 4 accumulators."""
    var acc0 = SIMD[DType.float32, NEON_WIDTH](0)
    var acc1 = SIMD[DType.float32, NEON_WIDTH](0)
    var acc2 = SIMD[DType.float32, NEON_WIDTH](0)
    var acc3 = SIMD[DType.float32, NEON_WIDTH](0)
    var k = 0

    # Process 4 blocks at a time
    var blk = 0
    while blk + 3 < nb:
        for blk_off in range(4):
            var b = blk + blk_off
            var blk_half = block.unsafe_offset(b * 18).unsafe_bitcast[Scalar[DType.float16]]()
            var d = Float32(blk_half.unsafe_load[width=1](offset=0))
            var qs = block.unsafe_offset(b * 18 + 2)

            # Load 16 bytes, split into two 8-element chunks
            var b0 = qs.unsafe_load[width=NEON_WIDTH](offset=0)
            var b1 = qs.unsafe_load[width=NEON_WIDTH](offset=NEON_WIDTH)

            # Low nibbles (first 8) - use FMA
            var lo0 = (b0 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
            var wv_lo0 = d * (lo0 - 8.0)
            var x0 = x.unsafe_load[width=NEON_WIDTH](offset=k + blk_off * 32).cast[DType.float32]()

            # Low nibbles (second 8)
            var lo1 = (b1 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
            var wv_lo1 = d * (lo1 - 8.0)
            var x1 = x.unsafe_load[width=NEON_WIDTH](offset=k + blk_off * 32 + NEON_WIDTH).cast[DType.float32]()

            # High nibbles (first 8)
            var hi0 = (b0 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
            var wv_hi0 = d * (hi0 - 8.0)
            var x2 = x.unsafe_load[width=NEON_WIDTH](offset=k + blk_off * 32 + 16).cast[DType.float32]()

            # High nibbles (second 8)
            var hi1 = (b1 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
            var wv_hi1 = d * (hi1 - 8.0)
            var x3 = x.unsafe_load[width=NEON_WIDTH](offset=k + blk_off * 32 + 16 + NEON_WIDTH).cast[DType.float32]()

            # Distribute across accumulators
            if blk_off == 0:
                acc0 = x0.fma(wv_lo0, acc0)
                acc0 = x1.fma(wv_lo1, acc0)
                acc0 = x2.fma(wv_hi0, acc0)
                acc0 = x3.fma(wv_hi1, acc0)
            elif blk_off == 1:
                acc1 = x0.fma(wv_lo0, acc1)
                acc1 = x1.fma(wv_lo1, acc1)
                acc1 = x2.fma(wv_hi0, acc1)
                acc1 = x3.fma(wv_hi1, acc1)
            elif blk_off == 2:
                acc2 = x0.fma(wv_lo0, acc2)
                acc2 = x1.fma(wv_lo1, acc2)
                acc2 = x2.fma(wv_hi0, acc2)
                acc2 = x3.fma(wv_hi1, acc2)
            else:
                acc3 = x0.fma(wv_lo0, acc3)
                acc3 = x1.fma(wv_lo1, acc3)
                acc3 = x2.fma(wv_hi0, acc3)
                acc3 = x3.fma(wv_hi1, acc3)

        k += 128  # 4 blocks * 32 elements
        blk += 4

    # Handle remaining blocks
    while blk < nb:
        var blk_half = block.unsafe_offset(blk * 18).unsafe_bitcast[Scalar[DType.float16]]()
        var d = Float32(blk_half.unsafe_load[width=1](offset=0))
        var qs = block.unsafe_offset(blk * 18 + 2)

        var b0 = qs.unsafe_load[width=NEON_WIDTH](offset=0)
        var b1 = qs.unsafe_load[width=NEON_WIDTH](offset=NEON_WIDTH)

        var lo0 = (b0 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
        var lo1 = (b1 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
        var x0 = x.unsafe_load[width=NEON_WIDTH](offset=k).cast[DType.float32]()
        var x1 = x.unsafe_load[width=NEON_WIDTH](offset=k + NEON_WIDTH).cast[DType.float32]()
        acc0 = x0.fma(d * (lo0 - 8.0), acc0)
        acc0 = x1.fma(d * (lo1 - 8.0), acc0)

        var hi0 = (b0 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
        var hi1 = (b1 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
        var x2 = x.unsafe_load[width=NEON_WIDTH](offset=k + 16).cast[DType.float32]()
        var x3 = x.unsafe_load[width=NEON_WIDTH](offset=k + 16 + NEON_WIDTH).cast[DType.float32]()
        acc0 = x2.fma(d * (hi0 - 8.0), acc0)
        acc0 = x3.fma(d * (hi1 - 8.0), acc0)

        k += 32
        blk += 1

    return (acc0 + acc1 + acc2 + acc3).reduce_add()


def vec_dot_q8_0_neon[
    dtype: DType
](
    x: Pointer[Scalar[dtype], MutUntrackedOrigin],
    block: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """NEON-optimized Q8_0 dot product with FMA and 4 accumulators."""
    var acc0 = SIMD[DType.float32, NEON_WIDTH](0)
    var acc1 = SIMD[DType.float32, NEON_WIDTH](0)
    var acc2 = SIMD[DType.float32, NEON_WIDTH](0)
    var acc3 = SIMD[DType.float32, NEON_WIDTH](0)
    var k = 0

    # Process 4 blocks at a time
    var blk = 0
    while blk + 3 < nb:
        for blk_off in range(4):
            var b = blk + blk_off
            var blk_half = block.unsafe_offset(b * 34).unsafe_bitcast[Scalar[DType.float16]]()
            var d = Float32(blk_half.unsafe_load[width=1](offset=0))
            var qs = block.unsafe_offset(b * 34 + 2)

            var acc_local = SIMD[DType.float32, NEON_WIDTH](0)
            for chunk in range(4):
                var q = qs.unsafe_load[width=NEON_WIDTH](offset=chunk * NEON_WIDTH)
                var q_f32 = q.cast[DType.int8]().cast[DType.float32]()
                var wv = d * q_f32
                var xv = x.unsafe_load[width=NEON_WIDTH](offset=k + blk_off * 32 + chunk * NEON_WIDTH).cast[DType.float32]()
                acc_local = xv.fma(wv, acc_local)

            if blk_off == 0:
                acc0 = acc0 + acc_local
            elif blk_off == 1:
                acc1 = acc1 + acc_local
            elif blk_off == 2:
                acc2 = acc2 + acc_local
            else:
                acc3 = acc3 + acc_local

        k += 128  # 4 blocks * 32 elements
        blk += 4

    # Handle remaining blocks
    while blk < nb:
        var blk_half = block.unsafe_offset(blk * 34).unsafe_bitcast[Scalar[DType.float16]]()
        var d = Float32(blk_half.unsafe_load[width=1](offset=0))
        var qs = block.unsafe_offset(blk * 34 + 2)

        for chunk in range(4):
            var q = qs.unsafe_load[width=NEON_WIDTH](offset=chunk * NEON_WIDTH)
            var q_f32 = q.cast[DType.int8]().cast[DType.float32]()
            var wv = d * q_f32
            var xv = x.unsafe_load[width=NEON_WIDTH](offset=k + chunk * NEON_WIDTH).cast[DType.float32]()
            acc0 = xv.fma(wv, acc0)

        k += 32
        blk += 1

    return (acc0 + acc1 + acc2 + acc3).reduce_add()

# -- Q4_K × Q8_K int8 dot product (llama.cpp approach) ------------------------
#
# This is the key optimization: quantize activations to Q8_K (int8) and use
# hardware int8 dot product (SDOT) instead of float32 SIMD.


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
    var d = Float32(w_half.unsafe_load[width=1](offset=0))
    var dmin = Float32(w_half.unsafe_load[width=1](offset=1))
    var scales = w_block.unsafe_offset(4)
    var qs = w_block.unsafe_offset(16)

    # Read Q8_K scale
    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())
    var q8_qs = q8_data.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    var q8_bsums = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()

    # Q4_K formula: value = d * scale[i] * q - dmin * min[i]
    # NO -8 offset (Q4_K uses unsigned 0-15, not centered around 0)
    # Bias: -dmin * q8_d * sum(min * sum_q8_int)
    var bias = Float32(0)
    for j in range(8):
        var (_, m) = _get_scale_min_k4(j, scales)
        var bs0 = Int32(q8_bsums.unsafe_offset(j * 2).unsafe_load())
        var bs1 = Int32(q8_bsums.unsafe_offset(j * 2 + 1).unsafe_load())
        # Bias from dmin * min term
        bias -= dmin * q8_d * Float32(m) * Float32(bs0 + bs1)

    # Main dot product using SDOT
    # Q4_K layout: 256 elements in 128 bytes, 32 elements per scale
    # Scale 0: elements 0-31, low nibbles of bytes 0-31
    # Scale 1: elements 32-63, high nibbles of bytes 0-31
    # Scale 2: elements 64-95, low nibbles of bytes 32-63
    # ...
    # Scale j: bytes (j//2)*32 to (j//2)*32+31, nibble = j%2
    var sumi = Int32(0)

    for j in range(8):
        var (sc, _) = _get_scale_min_k4(j, scales)

        # Byte offset for this scale
        var q4_byte_offset = (j // 2) * 32
        
        # Load 32 bytes of packed 4-bit values (will produce 32 elements)
        # We process in two 16-byte chunks
        var m4b = SIMD[DType.uint8, 16](0x0F)
        
        # First 16 elements
        var q4_bits0 = qs.unsafe_offset(q4_byte_offset).unsafe_load[width=16](offset=0)
        var q4_vals0: SIMD[DType.int8, 16]
        if j % 2 == 0:
            q4_vals0 = (q4_bits0 & m4b).cast[DType.int8]()
        else:
            q4_vals0 = (q4_bits0 >> SIMD[DType.uint8, 16](4)).cast[DType.int8]()
        
        var q8_offset_val = j * 32
        var q8_vals0 = q8_qs.unsafe_offset(q8_offset_val).unsafe_load[width=16](offset=0)
        var dot = neon_sdot(SIMD[DType.int32, 4](0), q4_vals0, q8_vals0)
        
        # Second 16 elements
        var q4_bits1 = qs.unsafe_offset(q4_byte_offset + 16).unsafe_load[width=16](offset=0)
        var q4_vals1: SIMD[DType.int8, 16]
        if j % 2 == 0:
            q4_vals1 = (q4_bits1 & m4b).cast[DType.int8]()
        else:
            q4_vals1 = (q4_bits1 >> SIMD[DType.uint8, 16](4)).cast[DType.int8]()
        
        var q8_vals1 = q8_qs.unsafe_offset(q8_offset_val + 16).unsafe_load[width=16](offset=0)
        dot = neon_sdot(dot, q4_vals1, q8_vals1)

        # Sum the 4 int32 lanes and apply scale
        var dot_sum = dot[0] + dot[1] + dot[2] + dot[3]
        sumi += dot_sum * Int32(sc)

    # Apply super-block scales and add bias
    return d * q8_d * Float32(sumi) + bias


# ============================================================================
# Q5_K × Q8_K dot product kernel
# ============================================================================

def vec_dot_q5_k_q8_k(
    # Q5_K weight block (176 bytes per 256 elements)
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    # Q8_K activation: scale at offset 0, int8 at offset 4, bsums at offset 260
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q5_K × Q8_K dot product following llama.cpp exactly.

    Q5_K block layout (176 bytes):
    - d: fp16 scale at offset 0
    - dmin: fp16 min scale at offset 2
    - scales: 12 bytes at offset 4
    - qh: 32 bytes at offset 16 (high bits, 1 bit per element)
    - qs: 128 bytes at offset 48 (low 4 bits, 256 elements packed)

    Q5_K value: 5-bit = low4 + (high_bit ? 16 : 0), range 0-31
    Bias: only dmin * min term (NO offset bias like Q4_K)
    """
    # Read Q5_K scales
    var w_half = w_block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(w_half.unsafe_load[width=1](offset=0))
    var dmin = Float32(w_half.unsafe_load[width=1](offset=1))
    var scales = w_block.unsafe_offset(4)
    var qh = w_block.unsafe_offset(16)
    var qs = w_block.unsafe_offset(48)

    # Read Q8_K scale
    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())
    var q8_qs = q8_data.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    var q8_bsums = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()

    # Compute bias: only dmin * min term (NO offset bias)
    var bias = Float32(0)
    for j in range(8):
        var (sc, m) = _get_scale_min_k4(j, scales)
        var bs0 = Int32(q8_bsums.unsafe_offset(j * 2).unsafe_load())
        var bs1 = Int32(q8_bsums.unsafe_offset(j * 2 + 1).unsafe_load())
        bias -= dmin * q8_d * Float32(m) * Float32(bs0 + bs1)

    # Main dot product - follow llama.cpp structure exactly
    # Process 256 elements: 4 groups of 64 elements
    # Each group: low nibbles (32) + high nibbles (32)
    # qh provides high bit, with mask m shifting for each 32-element batch
    var sumi = Int32(0)
    var m4b = SIMD[DType.uint8, 16](0x0F)

    # Following llama.cpp: j=0..3, each processes 64 elements
    # m starts at 1 and shifts left after each 32-element batch
    var q4_ptr = qs  # qs offset in the block
    var q8_ptr = q8_qs
    var m = UInt8(1)

    for j in range(4):
        # Low nibbles: 32 elements
        for l in range(32):
            var low4 = Int(q4_ptr.unsafe_load[width=1](offset=l)) & 0xF
            var hm_byte = Int(qh.unsafe_load[width=1](offset=l))
            if (hm_byte & Int(m)) != 0:
                low4 += 16
            var q8_val = Int32(q8_ptr.unsafe_offset(l).unsafe_load())
            # Scale is determined by element index
            var elem_idx = j * 64 + l
            var (sc, _) = _get_scale_min_k4(elem_idx // 32, scales)
            sumi += Int32(low4) * q8_val * Int32(sc)

        m = m << 1

        # High nibbles: 32 elements
        for l in range(32):
            var high4 = Int(q4_ptr.unsafe_load[width=1](offset=l)) >> 4
            var hm_byte = Int(qh.unsafe_load[width=1](offset=l))
            if (hm_byte & Int(m)) != 0:
                high4 += 16
            var q8_val = Int32(q8_ptr.unsafe_offset(32 + l).unsafe_load())
            var elem_idx = j * 64 + 32 + l
            var (sc, _) = _get_scale_min_k4(elem_idx // 32, scales)
            sumi += Int32(high4) * q8_val * Int32(sc)

        m = m << 1
        q4_ptr = q4_ptr.unsafe_offset(32)
        q8_ptr = q8_ptr.unsafe_offset(64)

    return d * q8_d * Float32(sumi) + bias


# ============================================================================
# Q6_K × Q8_K dot product kernel
# ============================================================================

def vec_dot_q6_k_q8_k(
    # Q6_K weight block (210 bytes per 256 elements)
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    # Q8_K activation: scale at offset 0, int8 at offset 4, bsums at offset 260
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q6_K × Q8_K dot product following llama.cpp exactly.

    Q6_K block layout (210 bytes):
    - ql: 128 bytes at offset 0 (lower 4 bits, 2 per byte)
    - qh: 64 bytes at offset 128 (upper 2 bits, 4 per byte)
    - scales: 16 bytes at offset 192 (int8 scales)
    - d: fp16 scale at offset 208

    Q6_K value: 6-bit = (low4 | (high2 << 4)) - 32
    NO dmin term (no bias from min)
    """
    var ql = w_block.unsafe_offset(0)
    var qh = w_block.unsafe_offset(128)
    var scales_ptr = w_block.unsafe_offset(192)
    var d = Float32(w_block.unsafe_offset(208).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())

    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())
    var q8_qs = q8_data.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()

    var sumi = Int32(0)

    # Follow llama.cpp structure: 2 groups of 128 elements
    var ql_ptr = ql
    var qh_ptr = qh
    var q8_ptr = q8_qs

    for outer in range(2):
        # Process 128 elements
        # Each l iteration produces 4 elements
        for l in range(32):
            var ql_val = Int(ql_ptr.unsafe_load[width=1](offset=l))
            var ql_val_hi = Int(ql_ptr.unsafe_load[width=1](offset=l + 32))
            var qh_val = Int(qh_ptr.unsafe_load[width=1](offset=l))

            # 4 elements from this iteration
            # a[l+0]: ql[l] low nibble | (qh[l] bits 0-1) << 4
            var q6_0 = ((ql_val & 0xF) | ((qh_val & 3) << 4)) - 32
            var sc_0 = Int32(scales_ptr.unsafe_load[width=1](offset=outer * 64 + l * 2 + 0))
            var q8_0 = Int32(q8_ptr.unsafe_offset(l + 0).unsafe_load())
            sumi += Int32(q6_0) * q8_0 * sc_0

            # a[l+32]: ql[l+32] low nibble | (qh[l] bits 2-3) << 4
            var q6_1 = ((ql_val_hi & 0xF) | (((qh_val >> 2) & 3) << 4)) - 32
            var sc_1 = Int32(scales_ptr.unsafe_load[width=1](offset=outer * 64 + l * 2 + 32))
            var q8_1 = Int32(q8_ptr.unsafe_offset(l + 32).unsafe_load())
            sumi += Int32(q6_1) * q8_1 * sc_1

            # a[l+64]: ql[l] high nibble | (qh[l] bits 4-5) << 4
            var q6_2 = ((ql_val >> 4) | (((qh_val >> 4) & 3) << 4)) - 32
            var sc_2 = Int32(scales_ptr.unsafe_load[width=1](offset=outer * 64 + l * 2 + 64))
            var q8_2 = Int32(q8_ptr.unsafe_offset(l + 64).unsafe_load())
            sumi += Int32(q6_2) * q8_2 * sc_2

            # a[l+96]: ql[l+32] high nibble | (qh[l] bits 6-7) << 4
            var q6_3 = ((ql_val_hi >> 4) | (((qh_val >> 6) & 3) << 4)) - 32
            var sc_3 = Int32(scales_ptr.unsafe_load[width=1](offset=outer * 64 + l * 2 + 96))
            var q8_3 = Int32(q8_ptr.unsafe_offset(l + 96).unsafe_load())
            sumi += Int32(q6_3) * q8_3 * sc_3

        ql_ptr = ql_ptr.unsafe_offset(64)
        qh_ptr = qh_ptr.unsafe_offset(32)
        q8_ptr = q8_ptr.unsafe_offset(128)

    return d * q8_d * Float32(sumi)


# ============================================================================
# Q2_K × Q8_K dot product kernel
# ============================================================================

def vec_dot_q2_k_q8_k(
    # Q2_K weight block (84 bytes per 256 elements)
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    # Q8_K activation: scale at offset 0, int8 at offset 4, bsums at offset 260
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q2_K × Q8_K dot product following llama.cpp exactly.

    Q2_K block layout (84 bytes):
    - scales[16]: at offset 0 (scale in low 4 bits, min in high 4 bits)
    - qs[64]: at offset 16 (2-bit values, 4 per byte, 256 elements total)
    - d: fp16 at offset 80
    - dmin: fp16 at offset 82

    Q2_K value: 2-bit, range 0-3
    Bias: dall * isum - dmin * summs
    """
    var scales = w_block.unsafe_offset(0)
    var qs = w_block.unsafe_offset(16)
    var d = Float32(w_block.unsafe_offset(80).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())
    var dmin = Float32(w_block.unsafe_offset(82).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())

    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())
    var q8_qs = q8_data.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    var q8_bsums = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()

    # Compute summs = sum of y[i].bsums[j] * (sc[j] >> 4)
    var summs = Int32(0)
    for j in range(16):
        var sc_val = Int(scales.unsafe_load[width=1](offset=j))
        var min_val = sc_val >> 4
        summs += Int32(q8_bsums.unsafe_offset(j).unsafe_load()) * Int32(min_val)

    # Main dot product following llama.cpp structure
    # Process 2 groups of 128 elements (k = 0, 1)
    # Each group: 4 sub-blocks of 32 elements
    var isum = Int32(0)
    var scale_idx = 0  # scale index
    var q2_ptr = qs
    var q8_ptr = q8_qs

    for k in range(2):  # QK_K/128 = 2
        var shift = 0
        for j in range(4):
            # First 16 elements
            var sc = Int(scales.unsafe_load[width=1](offset=scale_idx)) & 0xF
            scale_idx += 1
            var isuml = Int32(0)
            for l in range(16):
                var q2_val = Int(q2_ptr.unsafe_load[width=1](offset=l))
                var q2_bits = (q2_val >> shift) & 3
                var q8_val = Int32(q8_ptr.unsafe_offset(l).unsafe_load())
                isuml += q8_val * Int32(q2_bits)
            isum += Int32(sc) * isuml

            # Second 16 elements
            sc = Int(scales.unsafe_load[width=1](offset=scale_idx)) & 0xF
            scale_idx += 1
            isuml = Int32(0)
            for l in range(16):
                var q2_val = Int(q2_ptr.unsafe_load[width=1](offset=l))
                var q2_bits = (q2_val >> shift) & 3
                var q8_val = Int32(q8_ptr.unsafe_offset(16 + l).unsafe_load())
                isuml += q8_val * Int32(q2_bits)
            isum += Int32(sc) * isuml

            shift += 2
            q8_ptr = q8_ptr.unsafe_offset(32)

        q2_ptr = q2_ptr.unsafe_offset(32)

    # Final result: dall * isum - dmin * summs
    return d * q8_d * Float32(isum) - dmin * q8_d * Float32(summs)


# ============================================================================
# Q3_K × Q8_K dot product kernel
# ============================================================================

def vec_dot_q3_k_q8_k(
    # Q3_K weight block (110 bytes per 256 elements)
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    # Q8_K activation: scale at offset 0, int8 at offset 4, bsums at offset 260
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q3_K × Q8_K dot product following llama.cpp exactly.

    Q3_K block layout (110 bytes):
    - hmask[32]: at offset 0 (high bit for sign, 1 bit per element)
    - qs[64]: at offset 32 (2-bit low values, 4 per byte)
    - scales[12]: at offset 96 (6-bit scales)
    - d: fp16 at offset 108

    Q3_K value: 3-bit = low2 - (has_sign ? 0 : 4), range -4 to 3
    NO dmin term
    """
    var hmask = w_block.unsafe_offset(0)
    var qs = w_block.unsafe_offset(32)
    var scales_raw = w_block.unsafe_offset(96)
    var d = Float32(w_block.unsafe_offset(108).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())

    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())
    var q8_qs = q8_data.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()

    # Dequantize the 6-bit scales (following llama.cpp's complex unpacking)
    # auxs[0-3] from 12 bytes of scales
    var auxs_0 = Int(0)
    var auxs_1 = Int(0)
    var auxs_2 = Int(0)
    var auxs_3 = Int(0)
    
    # Load 12 bytes into auxs
    for b in range(4):
        auxs_0 |= Int(scales_raw.unsafe_load[width=1](offset=b)) << (b * 8)
    for b in range(4):
        auxs_1 |= Int(scales_raw.unsafe_load[width=1](offset=4 + b)) << (b * 8)
    for b in range(4):
        auxs_2 |= Int(scales_raw.unsafe_load[width=1](offset=8 + b)) << (b * 8)
    
    # Following llama.cpp: unpack scales
    # kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f
    var kmask1 = Int(0x03030303)
    var kmask2 = Int(0x0f0f0f0f)
    var tmp = auxs_2
    auxs_2 = ((auxs_0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    auxs_3 = ((auxs_1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    auxs_0 = (auxs_0 & kmask2) | (((tmp >> 0) & kmask1) << 4)
    auxs_1 = (auxs_1 & kmask2) | (((tmp >> 2) & kmask1) << 4)
    
    # auxs[0-3] now contain the unpacked 6-bit scales as int8s
    # We access them byte-by-byte using scales pointer

    var sumi = Int32(0)
    var q3_ptr = qs
    var hm_ptr = hmask
    var q8_ptr = q8_qs
    var m = UInt8(1)

    # Process 256 elements: 2 groups of 128 elements (j = 0, 128)
    for j in range(0, 256, 128):
        # For each 32-element batch, extract low 2 bits and apply sign
        for l in range(32):
            var q3_val = Int(q3_ptr.unsafe_load[width=1](offset=l))
            var hm_val = Int(hm_ptr.unsafe_load[width=1](offset=l))
            
            # 4 values from each byte: bits 0-1, 2-3, 4-5, 6-7
            # Each 32-element batch uses one bit of hmask
            for shift in range(0, 8, 2):
                var low2 = (q3_val >> shift) & 3
                var has_sign = (hm_val & Int(m)) != 0
                var q3: Int
                if has_sign:
                    q3 = low2
                else:
                    q3 = low2 - 4
                
                var idx = j + (shift // 2) * 32 + l
                # Get scale for this element
                var scale_idx_inner = idx // 16
                var scale_byte: Int
                if scale_idx_inner < 4:
                    scale_byte = (auxs_0 >> (scale_idx_inner * 8)) & 0xFF
                elif scale_idx_inner < 8:
                    scale_byte = (auxs_1 >> ((scale_idx_inner - 4) * 8)) & 0xFF
                elif scale_idx_inner < 12:
                    scale_byte = (auxs_2 >> ((scale_idx_inner - 8) * 8)) & 0xFF
                else:
                    scale_byte = (auxs_3 >> ((scale_idx_inner - 12) * 8)) & 0xFF
                
                var scale = Int32(scale_byte) - 32  # scales are offset by 32
                var q8_val = Int32(q8_ptr.unsafe_offset((shift // 2) * 32 + l).unsafe_load())
                sumi += Int32(q3) * q8_val * scale
            
            m = m << 1
        
        q3_ptr = q3_ptr.unsafe_offset(32)
        hm_ptr = hm_ptr.unsafe_offset(32)
        q8_ptr = q8_ptr.unsafe_offset(128)

    return d * q8_d * Float32(sumi)
