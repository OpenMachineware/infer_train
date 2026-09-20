# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# IQ4_XS × Q8_K vector dot product kernel
#
# IQ4_XS uses non-linear quantization with a lookup table (kvalues_iq4nl)
# for better precision at the same bit-width.
#
# Block structure (136 bytes for 256 weights):
# - d: FP16 super-block scale (2 bytes)
# - scales_h: uint16, high bits of scales (2 bytes)
# - scales_l[4]: uint8, low bits (4 bytes, each packs 2 4-bit scales)
# - qs[128]: uint8, 4-bit packed values (128 bytes)
#
# For each 64-element sub-block (32 qs bytes):
# - First 16 qs bytes → low nibbles + high nibbles = 32 values
# - These 32 values dot with first 32 Q8 values (scaled by ls1)
# - Next 16 qs bytes → low nibbles + high nibbles = 32 values
# - These 32 values dot with next 32 Q8 values (scaled by ls2)

from std.memory import Pointer
from std.origin import MutUntrackedOrigin


# Non-linear quantization values for IQ4 formats
# Maps 4-bit index to int8 value
comptime kvalues_iq4nl = SIMD[DType.int8, 16](
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
)


@always_inline
def _lookup_kvalue(idx: Int) -> Int:
    """Lookup value in kvalues_iq4nl table."""
    return Int(kvalues_iq4nl[idx])


def vec_dot_iq4xs_q8k(
    x: Pointer[UInt8, MutUntrackedOrigin],
    y: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """Compute dot product of IQ4_XS weight block × Q8_K activation block.

    x points to nb blocks of IQ4_XS (136 bytes each)
    y points to nb blocks of Q8_K (336 bytes each)

    Returns the dot product.
    """
    var sumf = Float32(0)

    # Process each super-block (256 weights)
    for ibl in range(nb):
        # Load super-block scale (FP16 at offset 0)
        var d_ptr = x.unsafe_offset(ibl * 136).unsafe_bitcast[Scalar[DType.float16]]()
        var d = Float32(d_ptr.unsafe_load[width=1](offset=0))

        # Load Q8_K scale (FP32 at offset 0)
        var y_d_ptr = y.unsafe_offset(ibl * 336).unsafe_bitcast[Scalar[DType.float32]]()
        var y_d = Float32(y_d_ptr.unsafe_load[width=1](offset=0))

        # Load scales_h (uint16 at offset 2)
        var scales_h_raw = x.unsafe_offset(ibl * 136 + 2)
        var h = UInt16(scales_h_raw.unsafe_load[width=1](offset=0)) |
                (UInt16(scales_h_raw.unsafe_load[width=1](offset=1)) << 8)

        var sumi1 = 0
        var sumi2 = 0

        # Process 4 sub-blocks (64 elements each)
        for ib in range(4):
            # Load scales_l[ib] (at offset 4 + ib)
            var scales_l = x.unsafe_load[width=1](offset=ibl * 136 + 4 + ib)

            # Decode scale for first 32 values
            var ls1 = Int(scales_l & 0xf) | Int((h << 4) & 0x30)
            ls1 = ls1 - 32

            # Decode scale for second 32 values
            var ls2 = Int(scales_l >> 4) | Int((h << 2) & 0x30)
            ls2 = ls2 - 32

            h = h >> 4

            # qs for this sub-block: 32 bytes at offset 8 + ib*32
            var q4_base = ibl * 136 + 8 + ib * 32
            # q8 for this sub-block: 64 bytes at offset 4 + ib*64
            var q8_base = ibl * 336 + 4 + ib * 64

            # Process first 16 qs bytes -> 32 values -> dot with first 32 Q8
            var prod1 = 0
            for j in range(16):
                var q4_byte = x.unsafe_load[width=1](offset=q4_base + j)

                # Low nibble
                var idx_lo = Int(q4_byte & 0xf)
                var qv_lo = _lookup_kvalue(idx_lo)

                # High nibble
                var idx_hi = Int((q4_byte >> 4) & 0xf)
                var qv_hi = _lookup_kvalue(idx_hi)

                # Load 2 Q8 values
                var q8_0 = y.unsafe_load[width=1](offset=q8_base + j * 2)
                var q8_1 = y.unsafe_load[width=1](offset=q8_base + j * 2 + 1)

                prod1 += qv_lo * Int(q8_0) + qv_hi * Int(q8_1)

            # Process next 16 qs bytes -> 32 values -> dot with next 32 Q8
            var prod2 = 0
            for j in range(16, 32):
                var q4_byte = x.unsafe_load[width=1](offset=q4_base + j)

                var idx_lo = Int(q4_byte & 0xf)
                var qv_lo = _lookup_kvalue(idx_lo)

                var idx_hi = Int((q4_byte >> 4) & 0xf)
                var qv_hi = _lookup_kvalue(idx_hi)

                var q8_0 = y.unsafe_load[width=1](offset=q8_base + j * 2)
                var q8_1 = y.unsafe_load[width=1](offset=q8_base + j * 2 + 1)

                prod2 += qv_lo * Int(q8_0) + qv_hi * Int(q8_1)

            sumi1 += prod1 * ls1
            sumi2 += prod2 * ls2

        sumf += d * y_d * Float32(sumi1 + sumi2)

    return sumf
