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
    """Optimized Q4_K × Q8_K dot product using NEON SDOT with unrolled loops.

    Q4_K block layout (144 bytes):
    - d: fp16 scale at offset 0
    - dmin: fp16 min scale at offset 2
    - scales: 12 bytes at offset 4
    - qs: 128 bytes at offset 16 (4-bit values, 256 elements packed)

    Q8_K layout (292 bytes):
    - d: float32 scale at offset 0
    - qs: 256 int8 at offset 4
    - bsums: 16 int16 at offset 260
    """
    # Read Q4_K scales
    var w_half = w_block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(w_half.unsafe_load[width=1](offset=0))
    var dmin = Float32(w_half.unsafe_load[width=1](offset=1))
    var scales = w_block.unsafe_offset(4)
    var qs = w_block.unsafe_offset(16)

    # Read Q8_K scale and data
    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())
    var q8_qs = q8_data.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    var q8_bsums = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()

    # Load all 128 bytes of Q4_K qs
    # Q4_K layout: 128 bytes hold 256 elements (2 per byte)
    # bytes 0-31: elements 0-31 (low nibble for scale 0) + 32-63 (high nibble for scale 1)
    # bytes 32-63: elements 64-95 (scale 2) + 96-127 (scale 3)
    # bytes 64-95: elements 128-159 (scale 4) + 160-191 (scale 5)
    # bytes 96-127: elements 192-223 (scale 6) + 224-255 (scale 7)
    # Load 32 bytes at a time (2 SIMD loads each)
    var m4b = SIMD[DType.uint8, 16](0x0F)
    var q4_b0_15 = qs.unsafe_load[width=16](offset=0)
    var q4_b16_31 = qs.unsafe_load[width=16](offset=16)
    var q4_b32_47 = qs.unsafe_load[width=16](offset=32)
    var q4_b48_63 = qs.unsafe_load[width=16](offset=48)
    var q4_b64_79 = qs.unsafe_load[width=16](offset=64)
    var q4_b80_95 = qs.unsafe_load[width=16](offset=80)
    var q4_b96_111 = qs.unsafe_load[width=16](offset=96)
    var q4_b112_127 = qs.unsafe_load[width=16](offset=112)

    # Unrolled: j=0 (bytes 0-31, low nibble -> elements 0-31)
    var (sc0, m0) = _get_scale_min_k4(0, scales)
    var q4_0_lo = (q4_b0_15 & m4b).cast[DType.int8]()
    var q4_0_hi = (q4_b16_31 & m4b).cast[DType.int8]()
    var q8_0 = q8_qs.unsafe_load[width=16](offset=0)
    var q8_16 = q8_qs.unsafe_load[width=16](offset=16)
    var dot_0 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q4_0_lo, q8_0), q4_0_hi, q8_16)

    # Unrolled: j=1 (bytes 0-31, high nibble -> elements 32-63)
    var (sc1, m1) = _get_scale_min_k4(1, scales)
    var q4_1_lo = (q4_b0_15 >> SIMD[DType.uint8, 16](4)).cast[DType.int8]()
    var q4_1_hi = (q4_b16_31 >> SIMD[DType.uint8, 16](4)).cast[DType.int8]()
    var q8_32 = q8_qs.unsafe_load[width=16](offset=32)
    var q8_48 = q8_qs.unsafe_load[width=16](offset=48)
    var dot_1 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q4_1_lo, q8_32), q4_1_hi, q8_48)

    # Unrolled: j=2 (bytes 32-63, low nibble -> elements 64-95)
    var (sc2, m2) = _get_scale_min_k4(2, scales)
    var q4_2_lo = (q4_b32_47 & m4b).cast[DType.int8]()
    var q4_2_hi = (q4_b48_63 & m4b).cast[DType.int8]()
    var q8_64 = q8_qs.unsafe_load[width=16](offset=64)
    var q8_80 = q8_qs.unsafe_load[width=16](offset=80)
    var dot_2 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q4_2_lo, q8_64), q4_2_hi, q8_80)

    # Unrolled: j=3 (bytes 32-63, high nibble -> elements 96-127)
    var (sc3, m3) = _get_scale_min_k4(3, scales)
    var q4_3_lo = (q4_b32_47 >> SIMD[DType.uint8, 16](4)).cast[DType.int8]()
    var q4_3_hi = (q4_b48_63 >> SIMD[DType.uint8, 16](4)).cast[DType.int8]()
    var q8_96 = q8_qs.unsafe_load[width=16](offset=96)
    var q8_112 = q8_qs.unsafe_load[width=16](offset=112)
    var dot_3 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q4_3_lo, q8_96), q4_3_hi, q8_112)

    # Unrolled: j=4 (bytes 64-95, low nibble -> elements 128-159)
    var (sc4, m4) = _get_scale_min_k4(4, scales)
    var q4_4_lo = (q4_b64_79 & m4b).cast[DType.int8]()
    var q4_4_hi = (q4_b80_95 & m4b).cast[DType.int8]()
    var q8_128 = q8_qs.unsafe_load[width=16](offset=128)
    var q8_144 = q8_qs.unsafe_load[width=16](offset=144)
    var dot_4 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q4_4_lo, q8_128), q4_4_hi, q8_144)

    # Unrolled: j=5 (bytes 64-95, high nibble -> elements 160-191)
    var (sc5, m5) = _get_scale_min_k4(5, scales)
    var q4_5_lo = (q4_b64_79 >> SIMD[DType.uint8, 16](4)).cast[DType.int8]()
    var q4_5_hi = (q4_b80_95 >> SIMD[DType.uint8, 16](4)).cast[DType.int8]()
    var q8_160 = q8_qs.unsafe_load[width=16](offset=160)
    var q8_176 = q8_qs.unsafe_load[width=16](offset=176)
    var dot_5 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q4_5_lo, q8_160), q4_5_hi, q8_176)

    # Unrolled: j=6 (bytes 96-127, low nibble -> elements 192-223)
    var (sc6, m6) = _get_scale_min_k4(6, scales)
    var q4_6_lo = (q4_b96_111 & m4b).cast[DType.int8]()
    var q4_6_hi = (q4_b112_127 & m4b).cast[DType.int8]()
    var q8_192 = q8_qs.unsafe_load[width=16](offset=192)
    var q8_208 = q8_qs.unsafe_load[width=16](offset=208)
    var dot_6 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q4_6_lo, q8_192), q4_6_hi, q8_208)

    # Unrolled: j=7 (bytes 96-127, high nibble -> elements 224-255)
    var (sc7, m7) = _get_scale_min_k4(7, scales)
    var q4_7_lo = (q4_b96_111 >> SIMD[DType.uint8, 16](4)).cast[DType.int8]()
    var q4_7_hi = (q4_b112_127 >> SIMD[DType.uint8, 16](4)).cast[DType.int8]()
    var q8_224 = q8_qs.unsafe_load[width=16](offset=224)
    var q8_240 = q8_qs.unsafe_load[width=16](offset=240)
    var dot_7 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q4_7_lo, q8_224), q4_7_hi, q8_240)

    # Sum up all dots with scales
    var sumi = (
        (dot_0[0] + dot_0[1] + dot_0[2] + dot_0[3]) * Int32(sc0) +
        (dot_1[0] + dot_1[1] + dot_1[2] + dot_1[3]) * Int32(sc1) +
        (dot_2[0] + dot_2[1] + dot_2[2] + dot_2[3]) * Int32(sc2) +
        (dot_3[0] + dot_3[1] + dot_3[2] + dot_3[3]) * Int32(sc3) +
        (dot_4[0] + dot_4[1] + dot_4[2] + dot_4[3]) * Int32(sc4) +
        (dot_5[0] + dot_5[1] + dot_5[2] + dot_5[3]) * Int32(sc5) +
        (dot_6[0] + dot_6[1] + dot_6[2] + dot_6[3]) * Int32(sc6) +
        (dot_7[0] + dot_7[1] + dot_7[2] + dot_7[3]) * Int32(sc7)
    )

    # Compute bias from dmin * min term
    var bias = Float32(0)
    # Unrolled bias computation
    var bs0 = Int32(q8_bsums.unsafe_load[width=1](offset=0))
    var bs1 = Int32(q8_bsums.unsafe_load[width=1](offset=1))
    bias -= dmin * q8_d * Float32(m0) * Float32(bs0 + bs1)
    bs0 = Int32(q8_bsums.unsafe_load[width=1](offset=2))
    bs1 = Int32(q8_bsums.unsafe_load[width=1](offset=3))
    bias -= dmin * q8_d * Float32(m1) * Float32(bs0 + bs1)
    bs0 = Int32(q8_bsums.unsafe_load[width=1](offset=4))
    bs1 = Int32(q8_bsums.unsafe_load[width=1](offset=5))
    bias -= dmin * q8_d * Float32(m2) * Float32(bs0 + bs1)
    bs0 = Int32(q8_bsums.unsafe_load[width=1](offset=6))
    bs1 = Int32(q8_bsums.unsafe_load[width=1](offset=7))
    bias -= dmin * q8_d * Float32(m3) * Float32(bs0 + bs1)
    bs0 = Int32(q8_bsums.unsafe_load[width=1](offset=8))
    bs1 = Int32(q8_bsums.unsafe_load[width=1](offset=9))
    bias -= dmin * q8_d * Float32(m4) * Float32(bs0 + bs1)
    bs0 = Int32(q8_bsums.unsafe_load[width=1](offset=10))
    bs1 = Int32(q8_bsums.unsafe_load[width=1](offset=11))
    bias -= dmin * q8_d * Float32(m5) * Float32(bs0 + bs1)
    bs0 = Int32(q8_bsums.unsafe_load[width=1](offset=12))
    bs1 = Int32(q8_bsums.unsafe_load[width=1](offset=13))
    bias -= dmin * q8_d * Float32(m6) * Float32(bs0 + bs1)
    bs0 = Int32(q8_bsums.unsafe_load[width=1](offset=14))
    bs1 = Int32(q8_bsums.unsafe_load[width=1](offset=15))
    bias -= dmin * q8_d * Float32(m7) * Float32(bs0 + bs1)

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
    """Q5_K × Q8_K dot product using NEON SIMD - follows llama.cpp ARM implementation.

    Q5_K block layout (176 bytes):
    - d: fp16 scale at offset 0
    - dmin: fp16 min scale at offset 2
    - scales: 12 bytes at offset 4
    - qh: 32 bytes at offset 16 (high bits, 1 bit per element, packed)
    - qs: 128 bytes at offset 48 (low 4 bits, 256 elements packed)

    Q5_K value: 5-bit = low4 + (high_bit ? 16 : 0), range 0-31
    """
    # Read Q5_K scales
    var w_half = w_block.unsafe_bitcast[Scalar[DType.float16]]()
    var d = Float32(w_half.unsafe_load[width=1](offset=0))
    var dmin = Float32(w_half.unsafe_load[width=1](offset=1))
    var scales_ptr = w_block.unsafe_offset(4)
    var qh = w_block.unsafe_offset(16)
    var qs = w_block.unsafe_offset(48)

    # Read Q8_K scale and data
    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())
    var q8_qs = q8_data.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    var q8_bsums = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()

    # Compute bias from dmin * min term
    var bias = Float32(0)
    for j in range(8):
        var (_, m) = _get_scale_min_k4(j, scales_ptr)
        var bs0 = Int32(q8_bsums.unsafe_offset(j * 2).unsafe_load())
        var bs1 = Int32(q8_bsums.unsafe_offset(j * 2 + 1).unsafe_load())
        bias -= dmin * q8_d * Float32(m) * Float32(bs0 + bs1)

    # Load all Q5_K data upfront for cache efficiency
    # qs: 128 bytes, qh: 32 bytes (high bits)
    var qhbits_0 = qh.unsafe_load[width=16](offset=0)
    var qhbits_1 = qh.unsafe_load[width=16](offset=16)
    
    # Load all 128 bytes of Q5_K qs
    var q5_b0_15 = qs.unsafe_load[width=16](offset=0)
    var q5_b16_31 = qs.unsafe_load[width=16](offset=16)
    var q5_b32_47 = qs.unsafe_load[width=16](offset=32)
    var q5_b48_63 = qs.unsafe_load[width=16](offset=48)
    var q5_b64_79 = qs.unsafe_load[width=16](offset=64)
    var q5_b80_95 = qs.unsafe_load[width=16](offset=80)
    var q5_b96_111 = qs.unsafe_load[width=16](offset=96)
    var q5_b112_127 = qs.unsafe_load[width=16](offset=112)
    
    # Load all 256 bytes of Q8_K qs
    var q8_0_15 = q8_qs.unsafe_load[width=16](offset=0)
    var q8_16_31 = q8_qs.unsafe_load[width=16](offset=16)
    var q8_32_47 = q8_qs.unsafe_load[width=16](offset=32)
    var q8_48_63 = q8_qs.unsafe_load[width=16](offset=48)
    var q8_64_79 = q8_qs.unsafe_load[width=16](offset=64)
    var q8_80_95 = q8_qs.unsafe_load[width=16](offset=80)
    var q8_96_111 = q8_qs.unsafe_load[width=16](offset=96)
    var q8_112_127 = q8_qs.unsafe_load[width=16](offset=112)
    var q8_128_143 = q8_qs.unsafe_load[width=16](offset=128)
    var q8_144_159 = q8_qs.unsafe_load[width=16](offset=144)
    var q8_160_175 = q8_qs.unsafe_load[width=16](offset=160)
    var q8_176_191 = q8_qs.unsafe_load[width=16](offset=176)
    var q8_192_207 = q8_qs.unsafe_load[width=16](offset=192)
    var q8_208_223 = q8_qs.unsafe_load[width=16](offset=208)
    var q8_224_239 = q8_qs.unsafe_load[width=16](offset=224)
    var q8_240_255 = q8_qs.unsafe_load[width=16](offset=240)
    
    var m4b = SIMD[DType.uint8, 16](0x0F)
    var mone = SIMD[DType.uint8, 16](1)
    var mtwo = SIMD[DType.uint8, 16](2)
    
    var sumi = Int32(0)
    
    # j=0: bytes 0-31 of qs, qh bits 0
    var (sc0, _) = _get_scale_min_k4(0, scales_ptr)
    var (sc1, _) = _get_scale_min_k4(1, scales_ptr)
    var q5h_0 = (qhbits_0 & mone) << SIMD[DType.uint8, 16](4)
    var q5h_1 = (qhbits_1 & mone) << SIMD[DType.uint8, 16](4)
    var q5h_2 = (qhbits_0 & mtwo) << SIMD[DType.uint8, 16](3)
    var q5h_3 = (qhbits_1 & mtwo) << SIMD[DType.uint8, 16](3)
    var q5_0 = ((q5_b0_15 & m4b) | q5h_0).cast[DType.int8]()
    var q5_1 = ((q5_b16_31 & m4b) | q5h_1).cast[DType.int8]()
    var q5_2 = ((q5_b0_15 >> SIMD[DType.uint8, 16](4)) | q5h_2).cast[DType.int8]()
    var q5_3 = ((q5_b16_31 >> SIMD[DType.uint8, 16](4)) | q5h_3).cast[DType.int8]()
    var dot_0 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q5_0, q8_0_15), q5_1, q8_16_31)
    var dot_1 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q5_2, q8_32_47), q5_3, q8_48_63)
    sumi += Int32(sc0) * dot_0.reduce_add() + Int32(sc1) * dot_1.reduce_add()
    
    # j=1: bytes 32-63 of qs, qh bits shifted by 2
    var (sc2, _) = _get_scale_min_k4(2, scales_ptr)
    var (sc3, _) = _get_scale_min_k4(3, scales_ptr)
    var qhbits_0_s2 = qhbits_0 >> SIMD[DType.uint8, 16](2)
    var qhbits_1_s2 = qhbits_1 >> SIMD[DType.uint8, 16](2)
    q5h_0 = (qhbits_0_s2 & mone) << SIMD[DType.uint8, 16](4)
    q5h_1 = (qhbits_1_s2 & mone) << SIMD[DType.uint8, 16](4)
    q5h_2 = (qhbits_0_s2 & mtwo) << SIMD[DType.uint8, 16](3)
    q5h_3 = (qhbits_1_s2 & mtwo) << SIMD[DType.uint8, 16](3)
    q5_0 = ((q5_b32_47 & m4b) | q5h_0).cast[DType.int8]()
    q5_1 = ((q5_b48_63 & m4b) | q5h_1).cast[DType.int8]()
    q5_2 = ((q5_b32_47 >> SIMD[DType.uint8, 16](4)) | q5h_2).cast[DType.int8]()
    q5_3 = ((q5_b48_63 >> SIMD[DType.uint8, 16](4)) | q5h_3).cast[DType.int8]()
    dot_0 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q5_0, q8_64_79), q5_1, q8_80_95)
    dot_1 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q5_2, q8_96_111), q5_3, q8_112_127)
    sumi += Int32(sc2) * dot_0.reduce_add() + Int32(sc3) * dot_1.reduce_add()
    
    # j=2: bytes 64-95 of qs, qh bits shifted by 4
    var (sc4, _) = _get_scale_min_k4(4, scales_ptr)
    var (sc5, _) = _get_scale_min_k4(5, scales_ptr)
    var qhbits_0_s4 = qhbits_0 >> SIMD[DType.uint8, 16](4)
    var qhbits_1_s4 = qhbits_1 >> SIMD[DType.uint8, 16](4)
    q5h_0 = (qhbits_0_s4 & mone) << SIMD[DType.uint8, 16](4)
    q5h_1 = (qhbits_1_s4 & mone) << SIMD[DType.uint8, 16](4)
    q5h_2 = (qhbits_0_s4 & mtwo) << SIMD[DType.uint8, 16](3)
    q5h_3 = (qhbits_1_s4 & mtwo) << SIMD[DType.uint8, 16](3)
    q5_0 = ((q5_b64_79 & m4b) | q5h_0).cast[DType.int8]()
    q5_1 = ((q5_b80_95 & m4b) | q5h_1).cast[DType.int8]()
    q5_2 = ((q5_b64_79 >> SIMD[DType.uint8, 16](4)) | q5h_2).cast[DType.int8]()
    q5_3 = ((q5_b80_95 >> SIMD[DType.uint8, 16](4)) | q5h_3).cast[DType.int8]()
    dot_0 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q5_0, q8_128_143), q5_1, q8_144_159)
    dot_1 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q5_2, q8_160_175), q5_3, q8_176_191)
    sumi += Int32(sc4) * dot_0.reduce_add() + Int32(sc5) * dot_1.reduce_add()
    
    # j=3: bytes 96-127 of qs, qh bits shifted by 6
    var (sc6, _) = _get_scale_min_k4(6, scales_ptr)
    var (sc7, _) = _get_scale_min_k4(7, scales_ptr)
    var qhbits_0_s6 = qhbits_0 >> SIMD[DType.uint8, 16](6)
    var qhbits_1_s6 = qhbits_1 >> SIMD[DType.uint8, 16](6)
    q5h_0 = (qhbits_0_s6 & mone) << SIMD[DType.uint8, 16](4)
    q5h_1 = (qhbits_1_s6 & mone) << SIMD[DType.uint8, 16](4)
    q5h_2 = (qhbits_0_s6 & mtwo) << SIMD[DType.uint8, 16](3)
    q5h_3 = (qhbits_1_s6 & mtwo) << SIMD[DType.uint8, 16](3)
    q5_0 = ((q5_b96_111 & m4b) | q5h_0).cast[DType.int8]()
    q5_1 = ((q5_b112_127 & m4b) | q5h_1).cast[DType.int8]()
    q5_2 = ((q5_b96_111 >> SIMD[DType.uint8, 16](4)) | q5h_2).cast[DType.int8]()
    q5_3 = ((q5_b112_127 >> SIMD[DType.uint8, 16](4)) | q5h_3).cast[DType.int8]()
    dot_0 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q5_0, q8_192_207), q5_1, q8_208_223)
    dot_1 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0), q5_2, q8_224_239), q5_3, q8_240_255)
    sumi += Int32(sc6) * dot_0.reduce_add() + Int32(sc7) * dot_1.reduce_add()
    
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
    """Q6_K × Q8_K dot product using NEON SIMD - optimized version.

    Q6_K block layout (210 bytes):
    - ql: 128 bytes at offset 0 (lower 4 bits, 2 per byte)
    - qh: 64 bytes at offset 128 (upper 2 bits, 4 per byte)
    - scales: 16 bytes at offset 192 (int8 scales, 16 total)
    - d: fp16 scale at offset 208

    Q6_K value: 6-bit = (low4 | (high2 << 4)) - 32, range -32 to 31
    NO dmin term (no bias from min)
    """
    var ql = w_block.unsafe_offset(0)
    var qh = w_block.unsafe_offset(128)
    var scales_ptr = w_block.unsafe_offset(192)
    var d = Float32(w_block.unsafe_offset(208).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())

    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())
    var q8_qs = q8_data.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()

    var m4b = SIMD[DType.uint8, 16](0x0F)
    var m2b = SIMD[DType.uint8, 16](3)
    var v32 = SIMD[DType.int8, 16](32)
    
    # Load all Q6_K data upfront
    # ql: 128 bytes total (64 bytes for outer=0, 64 bytes for outer=1)
    # qh: 64 bytes total (32 bytes for outer=0, 32 bytes for outer=1)
    var ql_0_15 = ql.unsafe_load[width=16](offset=0)
    var ql_16_31 = ql.unsafe_load[width=16](offset=16)
    var ql_32_47 = ql.unsafe_load[width=16](offset=32)
    var ql_48_63 = ql.unsafe_load[width=16](offset=48)
    var ql_64_79 = ql.unsafe_load[width=16](offset=64)
    var ql_80_95 = ql.unsafe_load[width=16](offset=80)
    var ql_96_111 = ql.unsafe_load[width=16](offset=96)
    var ql_112_127 = ql.unsafe_load[width=16](offset=112)
    
    var qh_0_15 = qh.unsafe_load[width=16](offset=0)
    var qh_16_31 = qh.unsafe_load[width=16](offset=16)
    var qh_32_47 = qh.unsafe_load[width=16](offset=32)
    var qh_48_63 = qh.unsafe_load[width=16](offset=48)
    
    # Load all 256 bytes of Q8_K qs
    var q8_0_15 = q8_qs.unsafe_load[width=16](offset=0)
    var q8_16_31 = q8_qs.unsafe_load[width=16](offset=16)
    var q8_32_47 = q8_qs.unsafe_load[width=16](offset=32)
    var q8_48_63 = q8_qs.unsafe_load[width=16](offset=48)
    var q8_64_79 = q8_qs.unsafe_load[width=16](offset=64)
    var q8_80_95 = q8_qs.unsafe_load[width=16](offset=80)
    var q8_96_111 = q8_qs.unsafe_load[width=16](offset=96)
    var q8_112_127 = q8_qs.unsafe_load[width=16](offset=112)
    var q8_128_143 = q8_qs.unsafe_load[width=16](offset=128)
    var q8_144_159 = q8_qs.unsafe_load[width=16](offset=144)
    var q8_160_175 = q8_qs.unsafe_load[width=16](offset=160)
    var q8_176_191 = q8_qs.unsafe_load[width=16](offset=176)
    var q8_192_207 = q8_qs.unsafe_load[width=16](offset=192)
    var q8_208_223 = q8_qs.unsafe_load[width=16](offset=208)
    var q8_224_239 = q8_qs.unsafe_load[width=16](offset=224)
    var q8_240_255 = q8_qs.unsafe_load[width=16](offset=240)
    
    # Load all 16 scales
    var scales_arr = scales_ptr.unsafe_load[width=16](offset=0)
    
    var sumi = Int32(0)
    
    # j=0: ql bytes 0-15, 16-31, 32-47, 48-63, qh bytes 0-15 and 16-31, scales 0-7
    # Following llama.cpp x86 exactly:
    # q4_0 = low(ql_0_15) | (qh_0_15 & 3) << 4
    # q4_1 = low(ql_16_31) | (qh_16_31 & 3) << 4
    # q4_2 = low(ql_32_47) | (qh_0_15 & 12) << 2
    # q4_3 = low(ql_48_63) | (qh_16_31 & 12) << 2
    # q4_4 = high(ql_0_15) | (qh_0_15 & 48)
    # q4_5 = high(ql_16_31) | (qh_16_31 & 48)
    # q4_6 = high(ql_32_47) | ((qh_0_15 & 192) >> 2)
    # q4_7 = high(ql_48_63) | ((qh_16_31 & 192) >> 2)
    
    var qh_0_bits = qh_0_15
    var qh_1_bits = qh_16_31
    
    var q6_0 = ((ql_0_15 & m4b) | ((qh_0_bits & m2b) << SIMD[DType.uint8, 16](4))).cast[DType.int8]() - v32
    var q6_1 = ((ql_16_31 & m4b) | ((qh_1_bits & m2b) << SIMD[DType.uint8, 16](4))).cast[DType.int8]() - v32
    var q6_2 = ((ql_32_47 & m4b) | ((qh_0_bits & SIMD[DType.uint8, 16](12)) << SIMD[DType.uint8, 16](2))).cast[DType.int8]() - v32
    var q6_3 = ((ql_48_63 & m4b) | ((qh_1_bits & SIMD[DType.uint8, 16](12)) << SIMD[DType.uint8, 16](2))).cast[DType.int8]() - v32
    var q6_4 = ((ql_0_15 >> SIMD[DType.uint8, 16](4)) | (qh_0_bits & SIMD[DType.uint8, 16](48))).cast[DType.int8]() - v32
    var q6_5 = ((ql_16_31 >> SIMD[DType.uint8, 16](4)) | (qh_1_bits & SIMD[DType.uint8, 16](48))).cast[DType.int8]() - v32
    var q6_6 = ((ql_32_47 >> SIMD[DType.uint8, 16](4)) | ((qh_0_bits & SIMD[DType.uint8, 16](192)) >> SIMD[DType.uint8, 16](2))).cast[DType.int8]() - v32
    var q6_7 = ((ql_48_63 >> SIMD[DType.uint8, 16](4)) | ((qh_1_bits & SIMD[DType.uint8, 16](192)) >> SIMD[DType.uint8, 16](2))).cast[DType.int8]() - v32
    
    var dot_0 = neon_sdot(SIMD[DType.int32, 4](0), q6_0, q8_0_15)
    var dot_1 = neon_sdot(SIMD[DType.int32, 4](0), q6_1, q8_16_31)
    var dot_2 = neon_sdot(SIMD[DType.int32, 4](0), q6_2, q8_32_47)
    var dot_3 = neon_sdot(SIMD[DType.int32, 4](0), q6_3, q8_48_63)
    var dot_4 = neon_sdot(SIMD[DType.int32, 4](0), q6_4, q8_64_79)
    var dot_5 = neon_sdot(SIMD[DType.int32, 4](0), q6_5, q8_80_95)
    var dot_6 = neon_sdot(SIMD[DType.int32, 4](0), q6_6, q8_96_111)
    var dot_7 = neon_sdot(SIMD[DType.int32, 4](0), q6_7, q8_112_127)
    
    sumi += Int32(scales_arr[0]) * dot_0.reduce_add()
    sumi += Int32(scales_arr[1]) * dot_1.reduce_add()
    sumi += Int32(scales_arr[2]) * dot_2.reduce_add()
    sumi += Int32(scales_arr[3]) * dot_3.reduce_add()
    sumi += Int32(scales_arr[4]) * dot_4.reduce_add()
    sumi += Int32(scales_arr[5]) * dot_5.reduce_add()
    sumi += Int32(scales_arr[6]) * dot_6.reduce_add()
    sumi += Int32(scales_arr[7]) * dot_7.reduce_add()
    
    # j=1: ql bytes 64-79, 80-95, 96-111, 112-127, qh bytes 32-47 and 48-63, scales 8-15
    qh_0_bits = qh_32_47
    qh_1_bits = qh_48_63
    
    q6_0 = ((ql_64_79 & m4b) | ((qh_0_bits & m2b) << SIMD[DType.uint8, 16](4))).cast[DType.int8]() - v32
    q6_1 = ((ql_80_95 & m4b) | ((qh_1_bits & m2b) << SIMD[DType.uint8, 16](4))).cast[DType.int8]() - v32
    q6_2 = ((ql_96_111 & m4b) | ((qh_0_bits & SIMD[DType.uint8, 16](12)) << SIMD[DType.uint8, 16](2))).cast[DType.int8]() - v32
    q6_3 = ((ql_112_127 & m4b) | ((qh_1_bits & SIMD[DType.uint8, 16](12)) << SIMD[DType.uint8, 16](2))).cast[DType.int8]() - v32
    q6_4 = ((ql_64_79 >> SIMD[DType.uint8, 16](4)) | (qh_0_bits & SIMD[DType.uint8, 16](48))).cast[DType.int8]() - v32
    q6_5 = ((ql_80_95 >> SIMD[DType.uint8, 16](4)) | (qh_1_bits & SIMD[DType.uint8, 16](48))).cast[DType.int8]() - v32
    q6_6 = ((ql_96_111 >> SIMD[DType.uint8, 16](4)) | ((qh_0_bits & SIMD[DType.uint8, 16](192)) >> SIMD[DType.uint8, 16](2))).cast[DType.int8]() - v32
    q6_7 = ((ql_112_127 >> SIMD[DType.uint8, 16](4)) | ((qh_1_bits & SIMD[DType.uint8, 16](192)) >> SIMD[DType.uint8, 16](2))).cast[DType.int8]() - v32
    
    dot_0 = neon_sdot(SIMD[DType.int32, 4](0), q6_0, q8_128_143)
    dot_1 = neon_sdot(SIMD[DType.int32, 4](0), q6_1, q8_144_159)
    dot_2 = neon_sdot(SIMD[DType.int32, 4](0), q6_2, q8_160_175)
    dot_3 = neon_sdot(SIMD[DType.int32, 4](0), q6_3, q8_176_191)
    dot_4 = neon_sdot(SIMD[DType.int32, 4](0), q6_4, q8_192_207)
    dot_5 = neon_sdot(SIMD[DType.int32, 4](0), q6_5, q8_208_223)
    dot_6 = neon_sdot(SIMD[DType.int32, 4](0), q6_6, q8_224_239)
    dot_7 = neon_sdot(SIMD[DType.int32, 4](0), q6_7, q8_240_255)
    
    sumi += Int32(scales_arr[8]) * dot_0.reduce_add()
    sumi += Int32(scales_arr[9]) * dot_1.reduce_add()
    sumi += Int32(scales_arr[10]) * dot_2.reduce_add()
    sumi += Int32(scales_arr[11]) * dot_3.reduce_add()
    sumi += Int32(scales_arr[12]) * dot_4.reduce_add()
    sumi += Int32(scales_arr[13]) * dot_5.reduce_add()
    sumi += Int32(scales_arr[14]) * dot_6.reduce_add()
    sumi += Int32(scales_arr[15]) * dot_7.reduce_add()

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
    """Q2_K × Q8_K dot product using NEON SIMD - optimized version.

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

    # Compute summs = sum of y[i].bsums[j] * (sc[j] >> 4) using SIMD
    # Load 16 scales (each byte has scale in low 4 bits, min in high 4 bits)
    var scales_vec = scales.unsafe_load[width=16](offset=0)
    # Load 16 bsums as int16
    var bsums_0_7 = q8_bsums.unsafe_load[width=8](offset=0).cast[DType.int32]()
    var bsums_8_15 = q8_bsums.unsafe_load[width=8](offset=8).cast[DType.int32]()
    
    # Extract mins (high 4 bits) and multiply with bsums
    var mins_vec = (scales_vec >> SIMD[DType.uint8, 16](4)).cast[DType.int32]()
    var summs = Int32(0)
    for i in range(8):
        summs += Int32(mins_vec[i]) * Int32(bsums_0_7[i])
    for i in range(8):
        summs += Int32(mins_vec[8 + i]) * Int32(bsums_8_15[i])

    # Load all 64 bytes of Q2_K qs upfront
    var q2_b0_15 = qs.unsafe_load[width=16](offset=0)
    var q2_b32_47 = qs.unsafe_load[width=16](offset=32)
    
    # Load all 256 bytes of Q8_K qs
    var q8_0_15 = q8_qs.unsafe_load[width=16](offset=0)
    var q8_16_31 = q8_qs.unsafe_load[width=16](offset=16)
    var q8_32_47 = q8_qs.unsafe_load[width=16](offset=32)
    var q8_48_63 = q8_qs.unsafe_load[width=16](offset=48)
    var q8_64_79 = q8_qs.unsafe_load[width=16](offset=64)
    var q8_80_95 = q8_qs.unsafe_load[width=16](offset=80)
    var q8_96_111 = q8_qs.unsafe_load[width=16](offset=96)
    var q8_112_127 = q8_qs.unsafe_load[width=16](offset=112)
    var q8_128_143 = q8_qs.unsafe_load[width=16](offset=128)
    var q8_144_159 = q8_qs.unsafe_load[width=16](offset=144)
    var q8_160_175 = q8_qs.unsafe_load[width=16](offset=160)
    var q8_176_191 = q8_qs.unsafe_load[width=16](offset=176)
    var q8_192_207 = q8_qs.unsafe_load[width=16](offset=192)
    var q8_208_223 = q8_qs.unsafe_load[width=16](offset=208)
    var q8_224_239 = q8_qs.unsafe_load[width=16](offset=224)
    var q8_240_255 = q8_qs.unsafe_load[width=16](offset=240)
    
    var m3 = SIMD[DType.uint8, 16](0x03)
    var isum = Int32(0)
    
    # k=0: q2 bytes 0-15, all 4 shifts (0, 2, 4, 6)
    # Each shift extracts 2 bits from the same 16 bytes, giving 16 elements
    # Each shift is used for TWO groups of 16 elements (same q2 bits, different q8)
    
    # Shift 0 (bits 0-1): scales 0, 1
    var q2_s0 = (q2_b0_15 & m3).cast[DType.int8]()
    var sc_0 = Int32(scales.unsafe_load[width=1](offset=0)) & 0xF
    var sc_1 = Int32(scales.unsafe_load[width=1](offset=1)) & 0xF
    var dot_s0_0 = neon_sdot(SIMD[DType.int32, 4](0), q2_s0, q8_0_15)
    var dot_s0_1 = neon_sdot(SIMD[DType.int32, 4](0), q2_s0, q8_16_31)
    isum += sc_0 * dot_s0_0.reduce_add() + sc_1 * dot_s0_1.reduce_add()
    
    # Shift 2 (bits 2-3): scales 2, 3
    var q2_s2 = ((q2_b0_15 >> SIMD[DType.uint8, 16](2)) & m3).cast[DType.int8]()
    var sc_2 = Int32(scales.unsafe_load[width=1](offset=2)) & 0xF
    var sc_3 = Int32(scales.unsafe_load[width=1](offset=3)) & 0xF
    var dot_s2_0 = neon_sdot(SIMD[DType.int32, 4](0), q2_s2, q8_32_47)
    var dot_s2_1 = neon_sdot(SIMD[DType.int32, 4](0), q2_s2, q8_48_63)
    isum += sc_2 * dot_s2_0.reduce_add() + sc_3 * dot_s2_1.reduce_add()
    
    # Shift 4 (bits 4-5): scales 4, 5
    var q2_s4 = ((q2_b0_15 >> SIMD[DType.uint8, 16](4)) & m3).cast[DType.int8]()
    var sc_4 = Int32(scales.unsafe_load[width=1](offset=4)) & 0xF
    var sc_5 = Int32(scales.unsafe_load[width=1](offset=5)) & 0xF
    var dot_s4_0 = neon_sdot(SIMD[DType.int32, 4](0), q2_s4, q8_64_79)
    var dot_s4_1 = neon_sdot(SIMD[DType.int32, 4](0), q2_s4, q8_80_95)
    isum += sc_4 * dot_s4_0.reduce_add() + sc_5 * dot_s4_1.reduce_add()
    
    # Shift 6 (bits 6-7): scales 6, 7
    var q2_s6 = ((q2_b0_15 >> SIMD[DType.uint8, 16](6)) & m3).cast[DType.int8]()
    var sc_6 = Int32(scales.unsafe_load[width=1](offset=6)) & 0xF
    var sc_7 = Int32(scales.unsafe_load[width=1](offset=7)) & 0xF
    var dot_s6_0 = neon_sdot(SIMD[DType.int32, 4](0), q2_s6, q8_96_111)
    var dot_s6_1 = neon_sdot(SIMD[DType.int32, 4](0), q2_s6, q8_112_127)
    isum += sc_6 * dot_s6_0.reduce_add() + sc_7 * dot_s6_1.reduce_add()
    
    # k=1: q2 bytes 32-47, all 4 shifts
    # Shift 0: scales 8, 9
    q2_s0 = (q2_b32_47 & m3).cast[DType.int8]()
    var sc_8 = Int32(scales.unsafe_load[width=1](offset=8)) & 0xF
    var sc_9 = Int32(scales.unsafe_load[width=1](offset=9)) & 0xF
    dot_s0_0 = neon_sdot(SIMD[DType.int32, 4](0), q2_s0, q8_128_143)
    dot_s0_1 = neon_sdot(SIMD[DType.int32, 4](0), q2_s0, q8_144_159)
    isum += sc_8 * dot_s0_0.reduce_add() + sc_9 * dot_s0_1.reduce_add()
    
    # Shift 2: scales 10, 11
    q2_s2 = ((q2_b32_47 >> SIMD[DType.uint8, 16](2)) & m3).cast[DType.int8]()
    var sc_10 = Int32(scales.unsafe_load[width=1](offset=10)) & 0xF
    var sc_11 = Int32(scales.unsafe_load[width=1](offset=11)) & 0xF
    dot_s2_0 = neon_sdot(SIMD[DType.int32, 4](0), q2_s2, q8_160_175)
    dot_s2_1 = neon_sdot(SIMD[DType.int32, 4](0), q2_s2, q8_176_191)
    isum += sc_10 * dot_s2_0.reduce_add() + sc_11 * dot_s2_1.reduce_add()
    
    # Shift 4: scales 12, 13
    q2_s4 = ((q2_b32_47 >> SIMD[DType.uint8, 16](4)) & m3).cast[DType.int8]()
    var sc_12 = Int32(scales.unsafe_load[width=1](offset=12)) & 0xF
    var sc_13 = Int32(scales.unsafe_load[width=1](offset=13)) & 0xF
    dot_s4_0 = neon_sdot(SIMD[DType.int32, 4](0), q2_s4, q8_192_207)
    dot_s4_1 = neon_sdot(SIMD[DType.int32, 4](0), q2_s4, q8_208_223)
    isum += sc_12 * dot_s4_0.reduce_add() + sc_13 * dot_s4_1.reduce_add()
    
    # Shift 6: scales 14, 15
    q2_s6 = ((q2_b32_47 >> SIMD[DType.uint8, 16](6)) & m3).cast[DType.int8]()
    var sc_14 = Int32(scales.unsafe_load[width=1](offset=14)) & 0xF
    var sc_15 = Int32(scales.unsafe_load[width=1](offset=15)) & 0xF
    dot_s6_0 = neon_sdot(SIMD[DType.int32, 4](0), q2_s6, q8_224_239)
    dot_s6_1 = neon_sdot(SIMD[DType.int32, 4](0), q2_s6, q8_240_255)
    isum += sc_14 * dot_s6_0.reduce_add() + sc_15 * dot_s6_1.reduce_add()

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
    """Q3_K × Q8_K dot product using NEON SIMD - optimized version.

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
    # This is scalar but only 16 values so it's fast
    var auxs_0 = Int(0)
    var auxs_1 = Int(0)
    var auxs_2 = Int(0)
    
    for b in range(4):
        auxs_0 |= Int(scales_raw.unsafe_load[width=1](offset=b)) << (b * 8)
    for b in range(4):
        auxs_1 |= Int(scales_raw.unsafe_load[width=1](offset=4 + b)) << (b * 8)
    for b in range(4):
        auxs_2 |= Int(scales_raw.unsafe_load[width=1](offset=8 + b)) << (b * 8)
    
    # Unpack scales
    var kmask1 = Int(0x03030303)
    var kmask2 = Int(0x0f0f0f0f)
    var tmp = auxs_2
    var auxs_2_unpacked = ((auxs_0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    var auxs_3_unpacked = ((auxs_1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    var auxs_0_unpacked = (auxs_0 & kmask2) | (((tmp >> 0) & kmask1) << 4)
    var auxs_1_unpacked = (auxs_1 & kmask2) | (((tmp >> 2) & kmask1) << 4)
    
    # Extract scales as int8 and subtract 32
    var scales_arr = SIMD[DType.int32, 16](0)
    for i in range(4):
        scales_arr[i] = Int32((auxs_0_unpacked >> (i * 8)) & 0xFF) - 32
    for i in range(4):
        scales_arr[4 + i] = Int32((auxs_1_unpacked >> (i * 8)) & 0xFF) - 32
    for i in range(4):
        scales_arr[8 + i] = Int32((auxs_2_unpacked >> (i * 8)) & 0xFF) - 32
    for i in range(4):
        scales_arr[12 + i] = Int32((auxs_3_unpacked >> (i * 8)) & 0xFF) - 32
    
    # Load all 32 bytes of hmask
    var hm_0_15 = hmask.unsafe_load[width=16](offset=0)
    var hm_16_31 = hmask.unsafe_load[width=16](offset=16)
    
    # Load all 64 bytes of qs
    var q3_b0_15 = qs.unsafe_load[width=16](offset=0)
    var q3_b16_31 = qs.unsafe_load[width=16](offset=16)
    var q3_b32_47 = qs.unsafe_load[width=16](offset=32)
    var q3_b48_63 = qs.unsafe_load[width=16](offset=48)
    
    # Load all 256 bytes of Q8_K qs
    var q8_0_15 = q8_qs.unsafe_load[width=16](offset=0)
    var q8_16_31 = q8_qs.unsafe_load[width=16](offset=16)
    var q8_32_47 = q8_qs.unsafe_load[width=16](offset=32)
    var q8_48_63 = q8_qs.unsafe_load[width=16](offset=48)
    var q8_64_79 = q8_qs.unsafe_load[width=16](offset=64)
    var q8_80_95 = q8_qs.unsafe_load[width=16](offset=80)
    var q8_96_111 = q8_qs.unsafe_load[width=16](offset=96)
    var q8_112_127 = q8_qs.unsafe_load[width=16](offset=112)
    var q8_128_143 = q8_qs.unsafe_load[width=16](offset=128)
    var q8_144_159 = q8_qs.unsafe_load[width=16](offset=144)
    var q8_160_175 = q8_qs.unsafe_load[width=16](offset=160)
    var q8_176_191 = q8_qs.unsafe_load[width=16](offset=176)
    var q8_192_207 = q8_qs.unsafe_load[width=16](offset=192)
    var q8_208_223 = q8_qs.unsafe_load[width=16](offset=208)
    var q8_224_239 = q8_qs.unsafe_load[width=16](offset=224)
    var q8_240_255 = q8_qs.unsafe_load[width=16](offset=240)
    
    var m3 = SIMD[DType.uint8, 16](0x03)
    var m0 = SIMD[DType.uint8, 16](0x01)
    var m1 = SIMD[DType.uint8, 16](0x02)
    var m2 = SIMD[DType.uint8, 16](0x04)
    var m3b = SIMD[DType.uint8, 16](0x08)
    var shift1 = SIMD[DType.uint8, 16](1)
    var shift2 = SIMD[DType.uint8, 16](2)
    
    var sumi = Int32(0)
    
    # Process using llama.cpp's algorithm:
    # Shift 0: q3h = (m0 & ~hmask) << 2 = (1 & ~hm) << 2 → 4 if bit 0 is 0
    # Shift 2: q3h = (m1 & ~hmask) << 1 = (2 & ~hm) << 1 → 4 if bit 1 is 0
    # Shift 4: q3h = (m2 & ~hmask) = (4 & ~hm) → 4 if bit 2 is 0
    # Shift 6: q3h = (m3 & ~hmask) >> 1 = (8 & ~hm) >> 1 → 4 if bit 3 is 0
    
    # j=0: use qs bytes 0-31, hmask bits 0-3
    # Shift 0: q3_b0_15 bits 0-1, hmask bit 0
    var low2_0 = (q3_b0_15 & m3).cast[DType.int8]()
    var q3h_0 = ((m0 & ~hm_0_15) << shift2).cast[DType.int8]()
    var q3_0 = low2_0 - q3h_0
    var sc_0 = scales_arr[0]
    var dot_0 = neon_sdot(SIMD[DType.int32, 4](0), q3_0, q8_0_15)
    sumi += sc_0 * dot_0.reduce_add()
    
    # Elements 16-31: q3_b16_31 bits 0-1, hmask bit 0
    var low2_1 = (q3_b16_31 & m3).cast[DType.int8]()
    var q3h_1 = ((m0 & ~hm_16_31) << shift2).cast[DType.int8]()
    var q3_1 = low2_1 - q3h_1
    var sc_1 = scales_arr[1]
    var dot_1 = neon_sdot(SIMD[DType.int32, 4](0), q3_1, q8_16_31)
    sumi += sc_1 * dot_1.reduce_add()
    
    # Shift 2: q3_b0_15 bits 2-3, hmask bit 1
    var low2_2 = ((q3_b0_15 >> shift2) & m3).cast[DType.int8]()
    var q3h_2 = ((m1 & ~hm_0_15) << shift1).cast[DType.int8]()
    var q3_2 = low2_2 - q3h_2
    var sc_2 = scales_arr[2]
    var dot_2 = neon_sdot(SIMD[DType.int32, 4](0), q3_2, q8_32_47)
    sumi += sc_2 * dot_2.reduce_add()
    
    # Elements 48-63: q3_b16_31 bits 2-3
    var low2_3 = ((q3_b16_31 >> shift2) & m3).cast[DType.int8]()
    var q3h_3 = ((m1 & ~hm_16_31) << shift1).cast[DType.int8]()
    var q3_3 = low2_3 - q3h_3
    var sc_3 = scales_arr[3]
    var dot_3 = neon_sdot(SIMD[DType.int32, 4](0), q3_3, q8_48_63)
    sumi += sc_3 * dot_3.reduce_add()
    
    # Shift 4: q3_b0_15 bits 4-5, hmask bit 2
    var low2_4 = ((q3_b0_15 >> SIMD[DType.uint8, 16](4)) & m3).cast[DType.int8]()
    var q3h_4 = (m2 & ~hm_0_15).cast[DType.int8]()
    var q3_4 = low2_4 - q3h_4
    var sc_4 = scales_arr[4]
    var dot_4 = neon_sdot(SIMD[DType.int32, 4](0), q3_4, q8_64_79)
    sumi += sc_4 * dot_4.reduce_add()
    
    # Elements 80-95: q3_b16_31 bits 4-5
    var low2_5 = ((q3_b16_31 >> SIMD[DType.uint8, 16](4)) & m3).cast[DType.int8]()
    var q3h_5 = (m2 & ~hm_16_31).cast[DType.int8]()
    var q3_5 = low2_5 - q3h_5
    var sc_5 = scales_arr[5]
    var dot_5 = neon_sdot(SIMD[DType.int32, 4](0), q3_5, q8_80_95)
    sumi += sc_5 * dot_5.reduce_add()
    
    # Shift 6: q3_b0_15 bits 6-7, hmask bit 3
    var low2_6 = ((q3_b0_15 >> SIMD[DType.uint8, 16](6)) & m3).cast[DType.int8]()
    var q3h_6 = ((m3b & ~hm_0_15) >> shift1).cast[DType.int8]()
    var q3_6 = low2_6 - q3h_6
    var sc_6 = scales_arr[6]
    var dot_6 = neon_sdot(SIMD[DType.int32, 4](0), q3_6, q8_96_111)
    sumi += sc_6 * dot_6.reduce_add()
    
    # Elements 112-127: q3_b16_31 bits 6-7
    var low2_7 = ((q3_b16_31 >> SIMD[DType.uint8, 16](6)) & m3).cast[DType.int8]()
    var q3h_7 = ((m3b & ~hm_16_31) >> shift1).cast[DType.int8]()
    var q3_7 = low2_7 - q3h_7
    var sc_7 = scales_arr[7]
    var dot_7 = neon_sdot(SIMD[DType.int32, 4](0), q3_7, q8_112_127)
    sumi += sc_7 * dot_7.reduce_add()
    
    # j=1: use qs bytes 32-63, hmask bits 0-3 (REUSED from j=0)
    # This matches llama.cpp SVE code which loads hmask once and reuses it
    # Shift 0: q3_b32_47 bits 0-1, hmask bit 0
    var low2_8 = (q3_b32_47 & m3).cast[DType.int8]()
    var q3h_8 = ((m0 & ~hm_0_15) << shift2).cast[DType.int8]()
    var q3_8 = low2_8 - q3h_8
    var sc_8 = scales_arr[8]
    var dot_8 = neon_sdot(SIMD[DType.int32, 4](0), q3_8, q8_128_143)
    sumi += sc_8 * dot_8.reduce_add()
    
    # Elements 144-159: q3_b48_63 bits 0-1, hmask bit 0
    var low2_9 = (q3_b48_63 & m3).cast[DType.int8]()
    var q3h_9 = ((m0 & ~hm_16_31) << shift2).cast[DType.int8]()
    var q3_9 = low2_9 - q3h_9
    var sc_9 = scales_arr[9]
    var dot_9 = neon_sdot(SIMD[DType.int32, 4](0), q3_9, q8_144_159)
    sumi += sc_9 * dot_9.reduce_add()
    
    # Shift 2: q3_b32_47 bits 2-3, hmask bit 1
    var low2_10 = ((q3_b32_47 >> shift2) & m3).cast[DType.int8]()
    var q3h_10 = ((m1 & ~hm_0_15) << shift1).cast[DType.int8]()
    var q3_10 = low2_10 - q3h_10
    var sc_10 = scales_arr[10]
    var dot_10 = neon_sdot(SIMD[DType.int32, 4](0), q3_10, q8_160_175)
    sumi += sc_10 * dot_10.reduce_add()
    
    # Elements 176-191: q3_b48_63 bits 2-3
    var low2_11 = ((q3_b48_63 >> shift2) & m3).cast[DType.int8]()
    var q3h_11 = ((m1 & ~hm_16_31) << shift1).cast[DType.int8]()
    var q3_11 = low2_11 - q3h_11
    var sc_11 = scales_arr[11]
    var dot_11 = neon_sdot(SIMD[DType.int32, 4](0), q3_11, q8_176_191)
    sumi += sc_11 * dot_11.reduce_add()
    
    # Shift 4: q3_b32_47 bits 4-5, hmask bit 2
    var low2_12 = ((q3_b32_47 >> SIMD[DType.uint8, 16](4)) & m3).cast[DType.int8]()
    var q3h_12 = (m2 & ~hm_0_15).cast[DType.int8]()
    var q3_12 = low2_12 - q3h_12
    var sc_12 = scales_arr[12]
    var dot_12 = neon_sdot(SIMD[DType.int32, 4](0), q3_12, q8_192_207)
    sumi += sc_12 * dot_12.reduce_add()
    
    # Elements 208-223: q3_b48_63 bits 4-5
    var low2_13 = ((q3_b48_63 >> SIMD[DType.uint8, 16](4)) & m3).cast[DType.int8]()
    var q3h_13 = (m2 & ~hm_16_31).cast[DType.int8]()
    var q3_13 = low2_13 - q3h_13
    var sc_13 = scales_arr[13]
    var dot_13 = neon_sdot(SIMD[DType.int32, 4](0), q3_13, q8_208_223)
    sumi += sc_13 * dot_13.reduce_add()
    
    # Shift 6: q3_b32_47 bits 6-7, hmask bit 3
    var low2_14 = ((q3_b32_47 >> SIMD[DType.uint8, 16](6)) & m3).cast[DType.int8]()
    var q3h_14 = ((m3b & ~hm_0_15) >> shift1).cast[DType.int8]()
    var q3_14 = low2_14 - q3h_14
    var sc_14 = scales_arr[14]
    var dot_14 = neon_sdot(SIMD[DType.int32, 4](0), q3_14, q8_224_239)
    sumi += sc_14 * dot_14.reduce_add()
    
    # Elements 240-255: q3_b48_63 bits 6-7
    var low2_15 = ((q3_b48_63 >> SIMD[DType.uint8, 16](6)) & m3).cast[DType.int8]()
    var q3h_15 = ((m3b & ~hm_16_31) >> shift1).cast[DType.int8]()
    var q3_15 = low2_15 - q3h_15
    var sc_15 = scales_arr[15]
    var dot_15 = neon_sdot(SIMD[DType.int32, 4](0), q3_15, q8_240_255)
    sumi += sc_15 * dot_15.reduce_add()

    return d * q8_d * Float32(sumi)
