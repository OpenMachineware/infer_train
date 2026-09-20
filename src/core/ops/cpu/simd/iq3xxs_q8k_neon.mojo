# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# IQ3_XXS × Q8_K kernel - NEON SIMD optimized version

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic
from std.builtin.globals import global_constant
from std.memory.unsafe import bitcast
from src.core.ops.cpu.simd.simd_neon import (
    neon_vreinterpretq_s8_u32, neon_vmulq_s8, neon_sdot, neon_addv,
)

comptime QK_K = 256

# IQ3_XXS grid as static constant (256 entries, stored in .rodata)
comptime IQ3XXS_GRID: Array[UInt32, 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
]

# keven_signs_q2xs lookup table stored as uint64 (128 entries)
# Each 8-byte entry encodes 8 signs (+1/-1) for one group
comptime KEVEN_SIGNS_Q2XS: Array[UInt64, 128] = [
    0x0101010101010101, 0xff010101010101ff, 0xff0101010101ff01, 0x010101010101ffff,
    0xff01010101ff0101, 0x0101010101ff01ff, 0x0101010101ffff01, 0xff01010101ffffff,
    0xff010101ff010101, 0x01010101ff0101ff, 0x01010101ff01ff01, 0xff010101ff01ffff,
    0x01010101ffff0101, 0xff010101ffff01ff, 0xff010101ffffff01, 0x01010101ffffffff,
    0xff0101ff01010101, 0x010101ff010101ff, 0x010101ff0101ff01, 0xff0101ff0101ffff,
    0x010101ff01ff0101, 0xff0101ff01ff01ff, 0xff0101ff01ffff01, 0x010101ff01ffffff,
    0x010101ffff010101, 0xff0101ffff0101ff, 0xff0101ffff01ff01, 0x010101ffff01ffff,
    0xff0101ffffff0101, 0x010101ffffff01ff, 0x010101ffffffff01, 0xff0101ffffffffff,
    0xff01ff0101010101, 0x0101ff01010101ff, 0x0101ff010101ff01, 0xff01ff010101ffff,
    0x0101ff0101ff0101, 0xff01ff0101ff01ff, 0xff01ff0101ffff01, 0x0101ff0101ffffff,
    0x0101ff01ff010101, 0xff01ff01ff0101ff, 0xff01ff01ff01ff01, 0x0101ff01ff01ffff,
    0xff01ff01ffff0101, 0x0101ff01ffff01ff, 0x0101ff01ffffff01, 0xff01ff01ffffffff,
    0x0101ffff01010101, 0xff01ffff010101ff, 0xff01ffff0101ff01, 0x0101ffff0101ffff,
    0xff01ffff01ff0101, 0x0101ffff01ff01ff, 0x0101ffff01ffff01, 0xff01ffff01ffffff,
    0xff01ffffff010101, 0x0101ffffff0101ff, 0x0101ffffff01ff01, 0xff01ffffff01ffff,
    0x0101ffffffff0101, 0xff01ffffffff01ff, 0xff01ffffffffff01, 0x0101ffffffffffff,
    0xffff010101010101, 0x01ff0101010101ff, 0x01ff01010101ff01, 0xffff01010101ffff,
    0x01ff010101ff0101, 0xffff010101ff01ff, 0xffff010101ffff01, 0x01ff010101ffffff,
    0x01ff0101ff010101, 0xffff0101ff0101ff, 0xffff0101ff01ff01, 0x01ff0101ff01ffff,
    0xffff0101ffff0101, 0x01ff0101ffff01ff, 0x01ff0101ffffff01, 0xffff0101ffffffff,
    0x01ff01ff01010101, 0xffff01ff010101ff, 0xffff01ff0101ff01, 0x01ff01ff0101ffff,
    0xffff01ff01ff0101, 0x01ff01ff01ff01ff, 0x01ff01ff01ffff01, 0xffff01ff01ffffff,
    0xffff01ffff010101, 0x01ff01ffff0101ff, 0x01ff01ffff01ff01, 0xffff01ffff01ffff,
    0x01ff01ffffff0101, 0xffff01ffffff01ff, 0xffff01ffffffff01, 0x01ff01ffffffffff,
    0x01ffff0101010101, 0xffffff01010101ff, 0xffffff010101ff01, 0x01ffff010101ffff,
    0xffffff0101ff0101, 0x01ffff0101ff01ff, 0x01ffff0101ffff01, 0xffffff0101ffffff,
    0xffffff01ff010101, 0x01ffff01ff0101ff, 0x01ffff01ff01ff01, 0xffffff01ff01ffff,
    0x01ffff01ffff0101, 0xffffff01ffff01ff, 0xffffff01ffffff01, 0x01ffff01ffffffff,
    0xffffffff01010101, 0x01ffffff010101ff, 0x01ffffff0101ff01, 0xffffffff0101ffff,
    0x01ffffff01ff0101, 0xffffffff01ff01ff, 0xffffffff01ffff01, 0x01ffffff01ffffff,
    0x01ffffffff010101, 0xffffffffff0101ff, 0xffffffffff01ff01, 0x01ffffffff01ffff,
    0xffffffffffff0101, 0x01ffffffffff01ff, 0x01ffffffffffff01, 0xffffffffffffffff,
]


# Struct for loading 4 vectors of int8
struct NeonS8x4(TrivialRegisterPassable):
    var val0: SIMD[DType.int8, 16]
    var val1: SIMD[DType.int8, 16]
    var val2: SIMD[DType.int8, 16]
    var val3: SIMD[DType.int8, 16]


@always_inline
def neon_ld1_s8_x4(ptr: Pointer[UInt8, MutUntrackedOrigin]) -> NeonS8x4:
    """Load 64 bytes using ld1.16b instruction (4 vectors)."""
    var ptr_s8 = ptr.unsafe_bitcast[Pointer[Int8, MutUntrackedOrigin]]()
    return llvm_intrinsic[
        "llvm.aarch64.neon.ld1x4.v16i8.p0i8", NeonS8x4, has_side_effect=True
    ](ptr_s8)


@always_inline
def pack_u32x4(w: UInt32, x: UInt32, y: UInt32, z: UInt32) -> SIMD[DType.uint32, 4]:
    """Pack 4 uint32 values into a SIMD vector (direct construction)."""
    return SIMD[DType.uint32, 4](w, x, y, z)


@always_inline
def combine_s8_from_u64(lo: UInt64, hi: UInt64) -> SIMD[DType.int8, 16]:
    """Combine two uint64 (int8x8) into int8x16."""
    var combined = SIMD[DType.uint64, 2](lo, hi)
    return bitcast[DType.int8, 16](combined)


def vec_dot_iq3xxs_q8k_neon(
    x: Pointer[UInt8, MutUntrackedOrigin],
    y: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """IQ3_XXS × Q8_K dot product - NEON SIMD optimized version.

    Block layout (98 bytes):
    - d: FP16 scale (2 bytes)
    - qs[0..63]: grid indices (64 bytes)
    - scales_and_signs[0..31]: packed scales and signs (32 bytes)
    """
    # Get reference to static grid and signs table
    ref grid_ref = global_constant[IQ3XXS_GRID]()
    var grid = grid_ref.unsafe_ptr()

    ref signs_ref = global_constant[KEVEN_SIGNS_Q2XS]()
    var signs64 = signs_ref.unsafe_ptr()

    var sumf = Float32(0)

    for i in range(nb):
        var x_base = i * 98
        var y_base = i * 292

        var d_x = Float32(x.unsafe_offset(x_base).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0))
        var d_y = Float32(y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_load[width=1](offset=0))
        var d = d_x * d_y

        var q3_ptr = x_base + 2  # qs starts at offset 2
        var gas_ptr = x_base + 66  # scales_and_signs starts at offset 66
        var q8_ptr = y_base + 4  # q8 starts after d (4 bytes)

        var sumf1 = Float32(0)
        var sumf2 = Float32(0)

        # Process 8 sub-blocks (32 elements each), 2 at a time
        for _ in range(4):  # 4 iterations of 2 blocks each
            # Load 64 bytes of Q8 values
            var q8b = neon_ld1_s8_x4(y.unsafe_offset(q8_ptr))
            q8_ptr += 64

            # Load scales_and_signs (2 x uint32) using memcpy-like load
            var gas0 = x.unsafe_load[width=4](offset=gas_ptr)
            var gas1 = x.unsafe_load[width=4](offset=gas_ptr + 4)
            gas_ptr += 8

            var aux32_0 = UInt32(gas0[0]) | (UInt32(gas0[1]) << 8) | (UInt32(gas0[2]) << 16) | (UInt32(gas0[3]) << 24)
            var aux32_1 = UInt32(gas1[0]) | (UInt32(gas1[1]) << 8) | (UInt32(gas1[2]) << 16) | (UInt32(gas1[3]) << 24)

            # Load 16 grid indices and create packed vectors
            var q3_vals = x.unsafe_load[width=16](offset=q3_ptr)
            q3_ptr += 16

            var aux32x4_0 = pack_u32x4(
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[0]))),
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[1]))),
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[2]))),
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[3]))),
            )
            var aux32x4_1 = pack_u32x4(
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[4]))),
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[5]))),
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[6]))),
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[7]))),
            )
            var aux32x4_2 = pack_u32x4(
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[8]))),
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[9]))),
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[10]))),
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[11]))),
            )
            var aux32x4_3 = pack_u32x4(
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[12]))),
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[13]))),
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[14]))),
                UInt32(grid.unsafe_load[width=1](offset=Int(q3_vals[15]))),
            )

            # Load signs from keven_signs_q2xs (like llama.cpp)
            # signs64 is a uint64 pointer, each entry is 8 bytes of signs
            var q3s_val0 = combine_s8_from_u64(
                signs64.unsafe_load[width=1](offset=Int((aux32_0 >> 0) & 127)),
                signs64.unsafe_load[width=1](offset=Int((aux32_0 >> 7) & 127)),
            )
            var q3s_val1 = combine_s8_from_u64(
                signs64.unsafe_load[width=1](offset=Int((aux32_0 >> 14) & 127)),
                signs64.unsafe_load[width=1](offset=Int((aux32_0 >> 21) & 127)),
            )
            var q3s_val2 = combine_s8_from_u64(
                signs64.unsafe_load[width=1](offset=Int((aux32_1 >> 0) & 127)),
                signs64.unsafe_load[width=1](offset=Int((aux32_1 >> 7) & 127)),
            )
            var q3s_val3 = combine_s8_from_u64(
                signs64.unsafe_load[width=1](offset=Int((aux32_1 >> 14) & 127)),
                signs64.unsafe_load[width=1](offset=Int((aux32_1 >> 21) & 127)),
            )

            # Multiply signs with grid values
            q3s_val0 = neon_vmulq_s8(q3s_val0, neon_vreinterpretq_s8_u32(aux32x4_0))
            q3s_val1 = neon_vmulq_s8(q3s_val1, neon_vreinterpretq_s8_u32(aux32x4_1))
            q3s_val2 = neon_vmulq_s8(q3s_val2, neon_vreinterpretq_s8_u32(aux32x4_2))
            q3s_val3 = neon_vmulq_s8(q3s_val3, neon_vreinterpretq_s8_u32(aux32x4_3))

            # SDOT dot products
            var p1 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0, 0, 0, 0), q3s_val0, q8b.val0), q3s_val1, q8b.val1)
            var p2 = neon_sdot(neon_sdot(SIMD[DType.int32, 4](0, 0, 0, 0), q3s_val2, q8b.val2), q3s_val3, q8b.val3)

            # Scale: 0.5 + (aux32 >> 28)
            var ls1 = Float32(0.5) + Float32(aux32_0 >> 28)
            var ls2 = Float32(0.5) + Float32(aux32_1 >> 28)

            sumf1 += Float32(neon_addv(p1)) * ls1
            sumf2 += Float32(neon_addv(p2)) * ls2

        # Final scale: multiply by d and 0.5 (from llama.cpp)
        sumf += d * (sumf1 + sumf2) * 0.5

    return sumf
