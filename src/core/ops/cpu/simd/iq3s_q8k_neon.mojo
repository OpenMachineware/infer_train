# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# IQ3_S × Q8_K kernel - NEON SIMD optimized version

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic
from src.core.ops.cpu.simd.simd_neon import (
    neon_vmovl_u8, neon_vmovl_high_u8, neon_vget_low_u8, neon_vget_high_u8,
    neon_vshlq_u16, neon_vandq_u16, neon_vorrq_u16, neon_vdupq_n_u16,
    neon_vceqq_u8, neon_vorrq_u8, neon_vreinterpretq_u8_u32, neon_vreinterpretq_s8_u32,
    neon_vreinterpretq_s8_u8, neon_vmulq_s8, neon_vcombine_u8, neon_sdot, neon_addv,
)

comptime QK_K = 256

# Structs for loading multiple vectors
struct NeonU8x2(TrivialRegisterPassable):
    var val0: SIMD[DType.uint8, 16]
    var val1: SIMD[DType.uint8, 16]

struct NeonS8x4(TrivialRegisterPassable):
    var val0: SIMD[DType.int8, 16]
    var val1: SIMD[DType.int8, 16]
    var val2: SIMD[DType.int8, 16]
    var val3: SIMD[DType.int8, 16]


@always_inline
def neon_ld1_u8_x2(ptr: Pointer[UInt8, MutUntrackedOrigin]) -> NeonU8x2:
    """Load 32 bytes using ld1.16b instruction."""
    return llvm_intrinsic[
        "llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonU8x2, has_side_effect=True
    ](ptr)


@always_inline
def neon_ld1_s8_x4(ptr: Pointer[UInt8, MutUntrackedOrigin]) -> NeonS8x4:
    """Load 64 bytes using ld1.16b instruction (4 vectors)."""
    var ptr_s8 = ptr.unsafe_bitcast[Pointer[Int8, MutUntrackedOrigin]]()
    return llvm_intrinsic[
        "llvm.aarch64.neon.ld1x4.v16i8.p0i8", NeonS8x4, has_side_effect=True
    ](ptr_s8)


@always_inline
def neon_tbl1(table: SIMD[DType.uint8, 16], indices: SIMD[DType.uint8, 16]) -> SIMD[DType.uint8, 16]:
    """NEON TBL1: parallel table lookup for 16 values."""
    return llvm_intrinsic[
        "llvm.aarch64.neon.tbl1.v16i8",
        SIMD[DType.uint8, 16],
        has_side_effect=False,
    ](table, indices)


def vec_dot_iq3s_q8k(
    x: Pointer[UInt8, MutUntrackedOrigin],
    y: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """IQ3_S × Q8_K dot product - exact match of ggml_vec_dot_iq3_s_q8_K_generic"""

    # IQ3_S grid (512 entries) - from llama.cpp
    var iq3s_grid: List[UInt32] = [
        0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
        0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
        0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
        0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
        0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
        0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
        0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
        0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
        0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
        0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
        0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
        0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
        0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
        0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
        0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
        0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
        0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
        0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
        0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
        0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
        0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
        0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
        0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
        0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
        0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
        0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
        0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
        0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
        0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
        0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
        0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
        0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
        0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
        0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
        0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
        0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
        0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
        0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
        0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
        0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
        0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
        0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
        0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
        0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
        0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
        0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
        0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
        0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
        0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
        0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
        0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
        0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
        0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
        0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
        0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
        0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
        0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
        0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
        0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
        0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
        0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
        0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
        0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
        0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
    ]

    var sumf = Float32(0)

    for i in range(nb):
        var x_base = i * 110
        var y_base = i * 292

        var d_x = Float32(x.unsafe_offset(x_base).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0))
        var d_y = Float32(y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_load[width=1](offset=0))
        var d = d_x * d_y

        # Load scales[4]
        var scales = [
            Int(x.unsafe_load[width=1](offset=x_base + 106 + 0)),
            Int(x.unsafe_load[width=1](offset=x_base + 106 + 1)),
            Int(x.unsafe_load[width=1](offset=x_base + 106 + 2)),
            Int(x.unsafe_load[width=1](offset=x_base + 106 + 3)),
        ]

        var bsum = 0
        var q8_offset = y_base + 4  # q8 starts after d (4 bytes)

        for ib32 in range(0, 8, 2):
            var scale_idx = ib32 // 2
            var ls1 = 2 * (scales[scale_idx] & 0xF) + 1
            var ls2 = 2 * (scales[scale_idx] >> 4) + 1

            var sumi = 0

            # First sub-block
            var qs_ptr = x_base + 2 + ib32 * 8
            var signs_ptr = x_base + 74 + ib32 * 4

            for l in range(4):
                var qs_val0 = Int(x.unsafe_load[width=1](offset=qs_ptr + 2*l + 0))
                var qs_val1 = Int(x.unsafe_load[width=1](offset=qs_ptr + 2*l + 1))
                var qh_val = Int(x.unsafe_load[width=1](offset=x_base + 66 + ib32))

                var idx1 = qs_val0 | ((qh_val << (8 - 2*l)) & 256)
                var idx2 = qs_val1 | ((qh_val << (7 - 2*l)) & 256)

                var grid1 = iq3s_grid[idx1]
                var grid2 = iq3s_grid[idx2]

                var sign_byte = Int(x.unsafe_load[width=1](offset=signs_ptr + l))

                for j in range(4):
                    var g1 = Int((grid1 >> UInt32(j * 8)) & 0xFF)
                    var g2 = Int((grid2 >> UInt32(j * 8)) & 0xFF)

                    if g1 > 127:
                        g1 = g1 - 256
                    if g2 > 127:
                        g2 = g2 - 256

                    # kmask_iq2xs = [1, 2, 4, 8, 16, 32, 64, 128]
                    # kmask[j] = 1 << j
                    if sign_byte & (1 << j):
                        g1 = -g1
                    if sign_byte & (1 << (j + 4)):
                        g2 = -g2

                    var q8_val1 = Int(y.unsafe_load[width=1](offset=q8_offset + j))
                    var q8_val2 = Int(y.unsafe_load[width=1](offset=q8_offset + j + 4))

                    if q8_val1 > 127:
                        q8_val1 = q8_val1 - 256
                    if q8_val2 > 127:
                        q8_val2 = q8_val2 - 256

                    sumi += g1 * q8_val1
                    sumi += g2 * q8_val2

                q8_offset += 8  # Move q8 pointer after each l iteration

            bsum += sumi * ls1

            # Second sub-block
            sumi = 0
            qs_ptr = x_base + 2 + (ib32 + 1) * 8
            signs_ptr = x_base + 74 + (ib32 + 1) * 4

            for l in range(4):
                var qs_val0 = Int(x.unsafe_load[width=1](offset=qs_ptr + 2*l + 0))
                var qs_val1 = Int(x.unsafe_load[width=1](offset=qs_ptr + 2*l + 1))
                var qh_val = Int(x.unsafe_load[width=1](offset=x_base + 66 + ib32 + 1))

                var idx1 = qs_val0 | ((qh_val << (8 - 2*l)) & 256)
                var idx2 = qs_val1 | ((qh_val << (7 - 2*l)) & 256)

                var grid1 = iq3s_grid[idx1]
                var grid2 = iq3s_grid[idx2]

                var sign_byte = Int(x.unsafe_load[width=1](offset=signs_ptr + l))

                for j in range(4):
                    var g1 = Int((grid1 >> UInt32(j * 8)) & 0xFF)
                    var g2 = Int((grid2 >> UInt32(j * 8)) & 0xFF)

                    if g1 > 127:
                        g1 = g1 - 256
                    if g2 > 127:
                        g2 = g2 - 256

                    # kmask_iq2xs = [1, 2, 4, 8, 16, 32, 64, 128]
                    # kmask[j] = 1 << j
                    if sign_byte & (1 << j):
                        g1 = -g1
                    if sign_byte & (1 << (j + 4)):
                        g2 = -g2

                    var q8_val1 = Int(y.unsafe_load[width=1](offset=q8_offset + j))
                    var q8_val2 = Int(y.unsafe_load[width=1](offset=q8_offset + j + 4))

                    if q8_val1 > 127:
                        q8_val1 = q8_val1 - 256
                    if q8_val2 > 127:
                        q8_val2 = q8_val2 - 256

                    sumi += g1 * q8_val1
                    sumi += g2 * q8_val2

                q8_offset += 8  # Move q8 pointer after each l iteration

            bsum += sumi * ls2

        sumf += d * Float32(bsum)

    return sumf


# ============================================================================
# NEON SIMD Optimized Version
# ============================================================================

# Precomputed masks for sign processing (from llama.cpp)
comptime k_mask1_0 = SIMD[DType.uint8, 16](
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
)

comptime k_mask1_1 = SIMD[DType.uint8, 16](
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03,
)

comptime k_mask2 = SIMD[DType.uint8, 16](
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
)

comptime hshift = SIMD[DType.int16, 8](8, 7, 6, 5, 4, 3, 2, 1)


def vec_dot_iq3s_q8k_neon(
    x: Pointer[UInt8, MutUntrackedOrigin],
    y: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """IQ3_S × Q8_K dot product - NEON SIMD optimized version.

    Follows llama.cpp's ggml_vec_dot_iq3_s_q8_K pattern:
    - SIMD index calculation using vmovl, vshlq, vorrq
    - TBL-based sign processing
    - SDOT for dot products
    """
    # IQ3_S grid (512 entries) - from llama.cpp
    var iq3s_grid: List[UInt32] = [
        0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
        0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
        0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
        0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
        0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
        0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
        0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
        0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
        0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
        0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
        0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
        0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
        0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
        0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
        0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
        0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
        0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
        0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
        0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
        0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
        0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
        0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
        0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
        0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
        0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
        0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
        0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
        0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
        0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
        0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
        0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
        0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
        0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
        0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
        0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
        0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
        0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
        0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
        0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
        0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
        0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
        0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
        0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
        0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
        0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
        0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
        0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
        0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
        0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
        0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
        0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
        0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
        0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
        0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
        0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
        0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
        0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
        0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
        0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
        0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
        0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
        0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
        0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
        0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
    ]

    var m256 = neon_vdupq_n_u16(256)
    var m1 = SIMD[DType.uint8, 16](1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    var sumf = Float32(0)

    for i in range(nb):
        var x_base = i * 110
        var y_base = i * 292

        var d_x = Float32(x.unsafe_offset(x_base).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0))
        var d_y = Float32(y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_load[width=1](offset=0))
        var d = d_x * d_y

        # Load scales
        var scales_raw = [
            x.unsafe_load[width=1](offset=x_base + 106 + 0),
            x.unsafe_load[width=1](offset=x_base + 106 + 1),
            x.unsafe_load[width=1](offset=x_base + 106 + 2),
            x.unsafe_load[width=1](offset=x_base + 106 + 3),
        ]

        var sumi1 = 0
        var sumi2 = 0

        var qs_ptr = x_base + 2
        var signs_ptr = x_base + 74
        var q8_ptr = y_base + 4
        var qh_ptr = x_base + 66

        # Process 8 sub-blocks (32 elements each), 2 at a time
        for ib32 in range(0, 8, 2):
            # Load 64 bytes of Q8 values
            var q8b = neon_ld1_s8_x4(y.unsafe_offset(q8_ptr))
            q8_ptr += 64

            # Load 16 bytes of qs
            var idx_l = x.unsafe_load[width=16](offset=qs_ptr)
            qs_ptr += 16

            # Process first sub-block (ib32)
            var qh_val0 = UInt16(x.unsafe_load[width=1](offset=qh_ptr + ib32))
            var qh_dup0 = neon_vdupq_n_u16(qh_val0)

            # Calculate indices: idx = vmovl_u8(qs_low) | ((qh << shift) & 256)
            var idx_low0 = neon_vmovl_u8(neon_vget_low_u8(idx_l))
            var qh_shifted0 = neon_vshlq_u16(hshift, qh_dup0)
            var qh_masked0 = neon_vandq_u16(qh_shifted0, m256)
            var indices0 = neon_vorrq_u16(idx_low0, qh_masked0)

            # Load 8 grid values using scalar loads (extract from SIMD vector)
            var aux0_0 = SIMD[DType.uint32, 4](
                iq3s_grid[Int(indices0[0])],
                iq3s_grid[Int(indices0[1])],
                iq3s_grid[Int(indices0[2])],
                iq3s_grid[Int(indices0[3])],
            )
            var aux0_1 = SIMD[DType.uint32, 4](
                iq3s_grid[Int(indices0[4])],
                iq3s_grid[Int(indices0[5])],
                iq3s_grid[Int(indices0[6])],
                iq3s_grid[Int(indices0[7])],
            )

            # Process second sub-block (ib32+1)
            var qh_val1 = UInt16(x.unsafe_load[width=1](offset=qh_ptr + ib32 + 1))
            var qh_dup1 = neon_vdupq_n_u16(qh_val1)

            var idx_high1 = neon_vmovl_high_u8(idx_l)
            var qh_shifted1 = neon_vshlq_u16(hshift, qh_dup1)
            var qh_masked1 = neon_vandq_u16(qh_shifted1, m256)
            var indices1 = neon_vorrq_u16(idx_high1, qh_masked1)

            var aux1_0 = SIMD[DType.uint32, 4](
                iq3s_grid[Int(indices1[0])],
                iq3s_grid[Int(indices1[1])],
                iq3s_grid[Int(indices1[2])],
                iq3s_grid[Int(indices1[3])],
            )
            var aux1_1 = SIMD[DType.uint32, 4](
                iq3s_grid[Int(indices1[4])],
                iq3s_grid[Int(indices1[5])],
                iq3s_grid[Int(indices1[6])],
                iq3s_grid[Int(indices1[7])],
            )

            # Load 4 sign bytes (2 for each sub-block)
            var sign_byte0 = x.unsafe_load[width=1](offset=signs_ptr)
            var sign_byte1 = x.unsafe_load[width=1](offset=signs_ptr + 1)
            var sign_byte2 = x.unsafe_load[width=1](offset=signs_ptr + 2)
            var sign_byte3 = x.unsafe_load[width=1](offset=signs_ptr + 3)
            signs_ptr += 4

            # Sign processing using TBL
            # First pair of sign bytes (for aux0_0 and aux0_1)
            var sign_pair0 = UInt32(sign_byte0) | (UInt32(sign_byte1) << 16)
            var vs0 = neon_vreinterpretq_u8_u32(SIMD[DType.uint32, 4](sign_pair0, sign_pair0, sign_pair0, sign_pair0))

            # TBL to spread sign bits
            var vs0_0 = neon_tbl1(vs0, k_mask1_0) & k_mask2
            var vs0_1 = neon_tbl1(vs0, k_mask1_1) & k_mask2

            # Compare with mask to get 0xFF where bit is set, then OR with 1 to get -1/+1
            vs0_0 = neon_vorrq_u8(neon_vceqq_u8(vs0_0, k_mask2), m1)
            vs0_1 = neon_vorrq_u8(neon_vceqq_u8(vs0_1, k_mask2), m1)

            # Multiply signs with grid values
            var q3s0 = neon_vmulq_s8(neon_vreinterpretq_s8_u8(vs0_0), neon_vreinterpretq_s8_u32(aux0_0))
            var q3s1 = neon_vmulq_s8(neon_vreinterpretq_s8_u8(vs0_1), neon_vreinterpretq_s8_u32(aux0_1))

            # Second pair of sign bytes (for aux1_0 and aux1_1)
            var sign_pair1 = UInt32(sign_byte2) | (UInt32(sign_byte3) << 16)
            var vs1 = neon_vreinterpretq_u8_u32(SIMD[DType.uint32, 4](sign_pair1, sign_pair1, sign_pair1, sign_pair1))

            var vs1_0 = neon_tbl1(vs1, k_mask1_0) & k_mask2
            var vs1_1 = neon_tbl1(vs1, k_mask1_1) & k_mask2

            vs1_0 = neon_vorrq_u8(neon_vceqq_u8(vs1_0, k_mask2), m1)
            vs1_1 = neon_vorrq_u8(neon_vceqq_u8(vs1_1, k_mask2), m1)

            var q3s2 = neon_vmulq_s8(neon_vreinterpretq_s8_u8(vs1_0), neon_vreinterpretq_s8_u32(aux1_0))
            var q3s3 = neon_vmulq_s8(neon_vreinterpretq_s8_u8(vs1_1), neon_vreinterpretq_s8_u32(aux1_1))

            # SDOT dot products
            var p1 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0, 0, 0, 0), q3s0, q8b.val0), q3s1, q8b.val1)
            var p2 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0, 0, 0, 0), q3s2, q8b.val2), q3s3, q8b.val3)

            # Apply scales and accumulate
            var ls0 = Int(scales_raw[ib32 // 2] & 0x0f) * 2 + 1
            var ls1 = Int(scales_raw[ib32 // 2] >> 4) * 2 + 1

            sumi1 += Int(neon_addv(p1)) * ls0
            sumi2 += Int(neon_addv(p2)) * ls1

        sumf += d * Float32(sumi1 + sumi2)

    return sumf
