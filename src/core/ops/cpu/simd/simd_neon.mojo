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

                # Load 16 bytes, split into two 8-element chunks
                var b0 = q.unsafe_load[width=NEON_WIDTH](offset=0)
                var b1 = q.unsafe_load[width=NEON_WIDTH](offset=NEON_WIDTH)

                # Low nibbles - use FMA
                var lo0 = (b0 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
                var lo1 = (b1 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
                var wv_lo0 = d0 * lo0 - m0v
                var wv_lo1 = d0 * lo1 - m0v
                var x0 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64).cast[DType.float32]()
                var x1 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + NEON_WIDTH).cast[DType.float32]()
                acc_local = x0.fma(wv_lo0, acc_local)
                acc_local = x1.fma(wv_lo1, acc_local)

                # High nibbles - use FMA
                var hi0 = (b0 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
                var hi1 = (b1 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
                var wv_hi0 = d1 * hi0 - m1v
                var wv_hi1 = d1 * hi1 - m1v
                var x2 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 32).cast[DType.float32]()
                var x3 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 32 + NEON_WIDTH).cast[DType.float32]()
                acc_local = x2.fma(wv_hi0, acc_local)
                acc_local = x3.fma(wv_hi1, acc_local)

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

            var lo0 = (b0 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
            var lo1 = (b1 & SIMD[DType.uint8, NEON_WIDTH](0x0F)).cast[DType.float32]()
            var x0 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64).cast[DType.float32]()
            var x1 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + NEON_WIDTH).cast[DType.float32]()
            acc0 = x0.fma(d0 * lo0 - m0v, acc0)
            acc0 = x1.fma(d0 * lo1 - m0v, acc0)

            var hi0 = (b0 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
            var hi1 = (b1 >> SIMD[DType.uint8, NEON_WIDTH](4)).cast[DType.float32]()
            var x2 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 32).cast[DType.float32]()
            var x3 = x.unsafe_load[width=NEON_WIDTH](offset=k + pair * 64 + 32 + NEON_WIDTH).cast[DType.float32]()
            acc0 = x2.fma(d1 * hi0 - m1v, acc0)
            acc0 = x3.fma(d1 * hi1 - m1v, acc0)

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