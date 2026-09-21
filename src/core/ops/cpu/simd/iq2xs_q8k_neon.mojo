# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# IQ2_XS × Q8_K kernel - NEON SIMD optimized version

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic
from std.builtin.globals import global_constant
from std.memory.unsafe import bitcast
from src.core.ops.cpu.simd.simd_neon import (
    neon_vmulq_s8, neon_sdot, neon_addv, neon_vpaddq_s32,
)

comptime QK_K = 256

# IQ2_XS grid as static constant (512 entries, stored in .rodata)
comptime IQ2XS_GRID: Array[UInt64, 512] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919,
    0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919,
    0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x080808082b080808,
    0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819,
    0x080808082b191908, 0x080808082b192b19, 0x080808082b2b0808, 0x0808081908080819,
    0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808,
    0x080808190819082b, 0x0808081908191919, 0x0808081908192b08, 0x0808081908192b2b,
    0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b,
    0x0808081919081919, 0x0808081919082b08, 0x0808081919190819, 0x0808081919191908,
    0x08080819192b0808, 0x08080819192b2b08, 0x080808192b080819, 0x080808192b081908,
    0x080808192b190808, 0x0808082b08080808, 0x0808082b0808082b, 0x0808082b08081919,
    0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808,
    0x0808082b19080819, 0x0808082b19081908, 0x0808082b19190808, 0x0808082b19191919,
    0x0808082b2b080808, 0x0808082b2b082b2b, 0x0808190808080819, 0x0808190808081908,
    0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b,
    0x0808190808191919, 0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908,
    0x0808190819080808, 0x080819081908082b, 0x0808190819081919, 0x0808190819082b08,
    0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x08081908192b0808,
    0x080819082b080819, 0x080819082b081908, 0x080819082b190808, 0x0808191908080808,
    0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908190819,
    0x0808191908191908, 0x08081919082b0808, 0x0808191919080819, 0x0808191919081908,
    0x0808191919190808, 0x08081919192b0819, 0x080819192b080808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b08190808, 0x0808192b082b192b, 0x0808192b19080808,
    0x0808192b1908082b, 0x0808192b2b081908, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808082b2b, 0x08082b0808190819,
    0x08082b0808191908, 0x08082b08082b0808, 0x08082b08082b1919, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b0819192b08, 0x08082b082b080808,
    0x08082b082b2b0808, 0x08082b082b2b2b2b, 0x08082b1908080819, 0x08082b1908081908,
    0x08082b1908190808, 0x08082b1919080808, 0x08082b192b080819, 0x08082b192b082b19,
    0x08082b2b08080808, 0x08082b2b082b0808, 0x08082b2b082b2b08, 0x08082b2b2b19192b,
    0x08082b2b2b2b0808, 0x0819080808080819, 0x0819080808081908, 0x081908080808192b,
    0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919,
    0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908, 0x0819080819080808,
    0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819,
    0x0819080819191908, 0x08190808192b0808, 0x08190808192b2b2b, 0x081908082b080819,
    0x081908082b081908, 0x081908082b190808, 0x0819081908080808, 0x081908190808082b,
    0x0819081908081919, 0x0819081908082b08, 0x0819081908190819, 0x0819081908191908,
    0x08190819082b0808, 0x0819081919080819, 0x0819081919081908, 0x0819081919190808,
    0x081908192b080808, 0x081908192b191908, 0x081908192b19192b, 0x0819082b08080819,
    0x0819082b08081908, 0x0819082b0808192b, 0x0819082b08190808, 0x0819082b19080808,
    0x0819082b192b0808, 0x0819190808080808, 0x081919080808082b, 0x0819190808081919,
    0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x08191908082b0808,
    0x0819190819080819, 0x0819190819081908, 0x0819190819082b19, 0x0819190819190808,
    0x08191908192b1908, 0x081919082b080808, 0x0819191908080819, 0x0819191908081908,
    0x0819191908190808, 0x0819191919080808, 0x0819192b08080808, 0x0819192b08191908,
    0x0819192b19082b19, 0x08192b0808080819, 0x08192b0808081908, 0x08192b0808190808,
    0x08192b080819082b, 0x08192b0819080808, 0x08192b0819191908, 0x08192b082b08192b,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b19192b192b, 0x08192b2b19190819,
    0x08192b2b2b2b2b19, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919,
    0x082b080808082b08, 0x082b080808082b2b, 0x082b080808190819, 0x082b080808191908,
    0x082b0808082b0808, 0x082b080819080819, 0x082b080819081908, 0x082b080819190808,
    0x082b08082b080808, 0x082b08082b2b0808, 0x082b081908080819, 0x082b081908081908,
    0x082b081908190808, 0x082b081919080808, 0x082b081919082b08, 0x082b0819192b1919,
    0x082b082b08080808, 0x082b082b082b082b, 0x082b082b2b080808, 0x082b082b2b2b2b08,
    0x082b190808080819, 0x082b190808081908, 0x082b190808190808, 0x082b1908082b2b19,
    0x082b190819080808, 0x082b191908080808, 0x082b191919080819, 0x082b19191919082b,
    0x082b19192b192b19, 0x082b192b08080819, 0x082b192b08192b2b, 0x082b192b2b2b192b,
    0x082b2b0808080808, 0x082b2b0808082b08, 0x082b2b0808082b2b, 0x082b2b08082b0808,
    0x082b2b0819191919, 0x082b2b082b082b08, 0x082b2b082b2b082b, 0x082b2b19192b2b08,
    0x082b2b192b190808, 0x082b2b2b08082b08, 0x082b2b2b082b0808, 0x082b2b2b2b08082b,
    0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819, 0x1908080808081908,
    0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b,
    0x1908080808191919, 0x1908080808192b08, 0x19080808082b0819, 0x19080808082b1908,
    0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08,
    0x1908080819082b2b, 0x1908080819190819, 0x1908080819191908, 0x19080808192b0808,
    0x19080808192b1919, 0x190808082b080819, 0x190808082b081908, 0x190808082b190808,
    0x1908081908080808, 0x190808190808082b, 0x1908081908081919, 0x1908081908082b08,
    0x1908081908190819, 0x1908081908191908, 0x19080819082b0808, 0x1908081919080819,
    0x1908081919081908, 0x1908081919190808, 0x190808192b080808, 0x190808192b081919,
    0x190808192b2b082b, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808,
    0x1908082b0819082b, 0x1908082b082b2b19, 0x1908082b19080808, 0x1908190808080808,
    0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808190819,
    0x1908190808191908, 0x1908190808192b19, 0x19081908082b0808, 0x1908190819080819,
    0x1908190819081908, 0x1908190819190808, 0x190819082b080808, 0x190819082b191908,
    0x1908191908080819, 0x1908191908081908, 0x1908191908190808, 0x19081919082b1908,
    0x1908191919080808, 0x190819192b192b2b, 0x1908192b08080808, 0x1908192b08082b2b,
    0x1908192b19081908, 0x1908192b19190808, 0x19082b0808080819, 0x19082b0808081908,
    0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919, 0x19082b0819191908,
    0x19082b08192b082b, 0x19082b1908080808, 0x19082b1908190819, 0x19082b1919081908,
    0x19082b1919190808, 0x19082b19192b2b19, 0x19082b2b08081908, 0x1919080808080808,
    0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819,
    0x1919080808191908, 0x19190808082b0808, 0x19190808082b2b08, 0x1919080819080819,
    0x1919080819081908, 0x1919080819190808, 0x191908082b080808, 0x1919081908080819,
    0x1919081908081908, 0x1919081908190808, 0x1919081908191919, 0x1919081919080808,
    0x191908191908082b, 0x1919082b08080808, 0x1919082b19081908, 0x1919082b2b2b2b2b,
    0x1919190808080819, 0x1919190808081908, 0x1919190808190808, 0x19191908082b0819,
    0x1919190819080808, 0x19191908192b0808, 0x191919082b080819, 0x191919082b2b0819,
    0x1919191908080808, 0x1919191908082b08, 0x191919192b080808, 0x191919192b082b08,
    0x1919192b082b0819, 0x1919192b192b2b08, 0x1919192b2b2b0819, 0x19192b0808080808,
    0x19192b0808191908, 0x19192b0819080819, 0x19192b0819190808, 0x19192b082b192b19,
    0x19192b1908192b2b, 0x19192b1919080808, 0x19192b191908082b, 0x19192b2b2b081919,
    0x192b080808080819, 0x192b080808081908, 0x192b080808190808, 0x192b080819080808,
    0x192b080819191908, 0x192b0808192b082b, 0x192b08082b08192b, 0x192b08082b2b2b19,
    0x192b081908080808, 0x192b082b082b1908, 0x192b082b19082b2b, 0x192b082b2b19082b,
    0x192b190808080808, 0x192b19080819192b, 0x192b191908190808, 0x192b191919080808,
    0x192b191919081919, 0x192b19192b2b1908, 0x192b2b0808080819, 0x192b2b08192b2b2b,
    0x192b2b19082b1919, 0x192b2b2b0808192b, 0x192b2b2b19191908, 0x192b2b2b192b082b,
    0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08,
    0x2b08080808190819, 0x2b08080808191908, 0x2b080808082b0808, 0x2b080808082b2b2b,
    0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808082b080808,
    0x2b0808082b08082b, 0x2b0808082b2b2b08, 0x2b0808082b2b2b2b, 0x2b08081908080819,
    0x2b08081908081908, 0x2b0808190808192b, 0x2b08081908190808, 0x2b08081919080808,
    0x2b08081919190819, 0x2b08081919192b19, 0x2b08082b08080808, 0x2b08082b082b0808,
    0x2b08082b2b080808, 0x2b08082b2b08082b, 0x2b08082b2b2b0808, 0x2b08082b2b2b2b08,
    0x2b08190808080819, 0x2b08190808081908, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190819080808, 0x2b081908192b0808, 0x2b0819082b082b19,
    0x2b08191908080808, 0x2b08191919081908, 0x2b0819192b2b1919, 0x2b08192b08192b08,
    0x2b08192b192b2b2b, 0x2b082b0808080808, 0x2b082b0808082b08, 0x2b082b08082b1919,
    0x2b082b0819192b2b, 0x2b082b082b080808, 0x2b082b082b08082b, 0x2b082b082b2b2b08,
    0x2b082b190808192b, 0x2b082b2b082b082b, 0x2b082b2b2b080808, 0x2b082b2b2b082b08,
    0x2b082b2b2b19192b, 0x2b082b2b2b2b2b08, 0x2b19080808080819, 0x2b19080808081908,
    0x2b19080808190808, 0x2b19080819080808, 0x2b1908081919192b, 0x2b1908082b081908,
    0x2b19081908080808, 0x2b190819082b082b, 0x2b190819192b1908, 0x2b19082b1919192b,
    0x2b19082b2b082b19, 0x2b19190808080808, 0x2b19190808081919, 0x2b19190819081908,
    0x2b19190819190808, 0x2b19190819192b08, 0x2b191919082b2b19, 0x2b1919192b190808,
    0x2b1919192b19082b, 0x2b19192b19080819, 0x2b192b0819190819, 0x2b192b082b2b192b,
    0x2b192b1919082b19, 0x2b192b2b08191919, 0x2b192b2b192b0808, 0x2b2b080808080808,
    0x2b2b08080808082b, 0x2b2b080808082b08, 0x2b2b080808082b2b, 0x2b2b0808082b0808,
    0x2b2b0808082b2b2b, 0x2b2b08082b2b0808, 0x2b2b081919190819, 0x2b2b081919192b19,
    0x2b2b08192b2b192b, 0x2b2b082b08080808, 0x2b2b082b0808082b, 0x2b2b082b08082b08,
    0x2b2b082b082b2b2b, 0x2b2b082b2b080808, 0x2b2b082b2b2b0808, 0x2b2b190819080808,
    0x2b2b19082b191919, 0x2b2b192b192b1919, 0x2b2b192b2b192b08, 0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808, 0x2b2b2b08082b082b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08, 0x2b2b2b1908081908, 0x2b2b2b192b081908, 0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08, 0x2b2b2b2b082b2b2b, 0x2b2b2b2b2b190819, 0x2b2b2b2b2b2b2b2b
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


# Struct for 4 int32x4 vectors (matches NEON int32x4x4_t)
struct NeonI32x4(TrivialRegisterPassable):
    var val0: SIMD[DType.int32, 4]
    var val1: SIMD[DType.int32, 4]
    var val2: SIMD[DType.int32, 4]
    var val3: SIMD[DType.int32, 4]

    def __init__(out self, v0: SIMD[DType.int32, 4], v1: SIMD[DType.int32, 4],
                  v2: SIMD[DType.int32, 4], v3: SIMD[DType.int32, 4]):
        self.val0 = v0
        self.val1 = v1
        self.val2 = v2
        self.val3 = v3


@always_inline
def neon_ld1_s8_x4(ptr: Pointer[UInt8, MutUntrackedOrigin]) -> NeonS8x4:
    """Load 64 bytes using ld1.16b instruction (4 vectors)."""
    var ptr_s8 = ptr.unsafe_bitcast[Pointer[Int8, MutUntrackedOrigin]]()
    return llvm_intrinsic[
        "llvm.aarch64.neon.ld1x4.v16i8.p0i8", NeonS8x4, has_side_effect=True
    ](ptr_s8)


@always_inline
def combine_s8_from_u64(lo: UInt64, hi: UInt64) -> SIMD[DType.int8, 16]:
    """Combine two uint64 (int8x8) into int8x16."""
    var combined = SIMD[DType.uint64, 2](lo, hi)
    return bitcast[DType.int8, 16](combined)


def vec_dot_iq2xs_q8k_neon(
    x: Pointer[UInt8, MutUntrackedOrigin],
    y: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """IQ2_XS × Q8_K dot product - NEON SIMD optimized version.

    Block layout:
    - IQ2_XS: 74 bytes (d: 2, qs: 64, scales: 8)
    - Q8_K: 292 bytes (d: 4, qs: 256, bsums: 32)

    Grid: 512 entries
    Sign table: keven_signs_q2xs (128 entries)
    """
    # Get reference to static grid and signs table
    ref grid_ref = global_constant[IQ2XS_GRID]()
    var grid = grid_ref.unsafe_ptr()

    ref signs_ref = global_constant[KEVEN_SIGNS_Q2XS]()
    var signs64 = signs_ref.unsafe_ptr()

    var sumf = Float32(0)

    for i in range(nb):
        var x_base = i * 74
        var y_base = i * 292

        var d_x = Float32(x.unsafe_offset(x_base).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0))
        var d_y = Float32(y.unsafe_offset(y_base).unsafe_bitcast[Scalar[DType.float32]]().unsafe_load[width=1](offset=0))
        var d = d_x * d_y

        var q2_ptr = x_base + 2  # qs starts at offset 2
        var scales_ptr = x_base + 66  # scales starts at offset 66
        var q8_ptr = y_base + 4  # q8 starts after d (4 bytes)

        # Process scales: load 8 bytes, split into 16 nibbles - VECTORIZED
        # Use vectorized load instead of scalar loop
        var scales8_raw = x.unsafe_load[width=8](offset=scales_ptr)

        # Vectorized nibble extraction
        var mask4 = SIMD[DType.uint8, 8](0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF, 0xF)
        var scales_l = scales8_raw & mask4  # Low nibbles
        var scales_h = scales8_raw >> 4     # High nibbles

        # Zip: interleave l and h to get [l0, h0, l1, h1, ...]
        var combined = SIMD[DType.uint8, 16]()
        combined[0] = scales_l[0]
        combined[1] = scales_h[0]
        combined[2] = scales_l[1]
        combined[3] = scales_h[1]
        combined[4] = scales_l[2]
        combined[5] = scales_h[2]
        combined[6] = scales_l[3]
        combined[7] = scales_h[3]
        combined[8] = scales_l[4]
        combined[9] = scales_h[4]
        combined[10] = scales_l[5]
        combined[11] = scales_h[5]
        combined[12] = scales_l[6]
        combined[13] = scales_h[6]
        combined[14] = scales_l[7]
        combined[15] = scales_h[7]

        # Apply formula: 2 * scale + 1
        combined = combined * UInt8(2) + UInt8(1)

        # Vectorized conversion to int32x4 vectors
        var scales32 = NeonI32x4(
            SIMD[DType.int32, 4](Int32(combined[0]), Int32(combined[1]), Int32(combined[2]), Int32(combined[3])),
            SIMD[DType.int32, 4](Int32(combined[4]), Int32(combined[5]), Int32(combined[6]), Int32(combined[7])),
            SIMD[DType.int32, 4](Int32(combined[8]), Int32(combined[9]), Int32(combined[10]), Int32(combined[11])),
            SIMD[DType.int32, 4](Int32(combined[12]), Int32(combined[13]), Int32(combined[14]), Int32(combined[15])),
        )

        var sumi = SIMD[DType.int32, 4](0, 0, 0, 0)

        # Process 4 sub-blocks (64 elements each)
        for ib64 in range(4):
            # Load 64 bytes of Q8 values
            var q8b = neon_ld1_s8_x4(y.unsafe_offset(q8_ptr))
            q8_ptr += 64

            # Load 8 uint16 values (16 bytes) - direct byte extraction
            var qs_bytes = x.unsafe_load[width=16](offset=q2_ptr)
            q2_ptr += 16

            # Extract uint16 values using direct byte operations
            var q2_0 = UInt16(qs_bytes[0]) | (UInt16(qs_bytes[1]) << 8)
            var q2_1 = UInt16(qs_bytes[2]) | (UInt16(qs_bytes[3]) << 8)
            var q2_2 = UInt16(qs_bytes[4]) | (UInt16(qs_bytes[5]) << 8)
            var q2_3 = UInt16(qs_bytes[6]) | (UInt16(qs_bytes[7]) << 8)
            var q2_4 = UInt16(qs_bytes[8]) | (UInt16(qs_bytes[9]) << 8)
            var q2_5 = UInt16(qs_bytes[10]) | (UInt16(qs_bytes[11]) << 8)
            var q2_6 = UInt16(qs_bytes[12]) | (UInt16(qs_bytes[13]) << 8)
            var q2_7 = UInt16(qs_bytes[14]) | (UInt16(qs_bytes[15]) << 8)

            # Load grid values (grid index = q2_vals[j] & 511, sign index = q2_vals[j] >> 9)
            var q2u_0 = combine_s8_from_u64(
                grid.unsafe_load[width=1](offset=Int(q2_0 & 511)),
                grid.unsafe_load[width=1](offset=Int(q2_1 & 511)),
            )
            var q2u_1 = combine_s8_from_u64(
                grid.unsafe_load[width=1](offset=Int(q2_2 & 511)),
                grid.unsafe_load[width=1](offset=Int(q2_3 & 511)),
            )
            var q2u_2 = combine_s8_from_u64(
                grid.unsafe_load[width=1](offset=Int(q2_4 & 511)),
                grid.unsafe_load[width=1](offset=Int(q2_5 & 511)),
            )
            var q2u_3 = combine_s8_from_u64(
                grid.unsafe_load[width=1](offset=Int(q2_6 & 511)),
                grid.unsafe_load[width=1](offset=Int(q2_7 & 511)),
            )

            # Load signs (sign index = q2_vals[j] >> 9, max 127)
            var q2s_0 = combine_s8_from_u64(
                signs64.unsafe_load[width=1](offset=Int(q2_0 >> 9)),
                signs64.unsafe_load[width=1](offset=Int(q2_1 >> 9)),
            )
            var q2s_1 = combine_s8_from_u64(
                signs64.unsafe_load[width=1](offset=Int(q2_2 >> 9)),
                signs64.unsafe_load[width=1](offset=Int(q2_3 >> 9)),
            )
            var q2s_2 = combine_s8_from_u64(
                signs64.unsafe_load[width=1](offset=Int(q2_4 >> 9)),
                signs64.unsafe_load[width=1](offset=Int(q2_5 >> 9)),
            )
            var q2s_3 = combine_s8_from_u64(
                signs64.unsafe_load[width=1](offset=Int(q2_6 >> 9)),
                signs64.unsafe_load[width=1](offset=Int(q2_7 >> 9)),
            )

            # Multiply signs with grid values
            q2u_0 = neon_vmulq_s8(q2u_0, q2s_0)
            q2u_1 = neon_vmulq_s8(q2u_1, q2s_1)
            q2u_2 = neon_vmulq_s8(q2u_2, q2s_2)
            q2u_3 = neon_vmulq_s8(q2u_3, q2s_3)

            # SDOT dot products
            var p1 = neon_sdot(SIMD[DType.int32, 4](0, 0, 0, 0), q2u_0, q8b.val0)
            var p2 = neon_sdot(SIMD[DType.int32, 4](0, 0, 0, 0), q2u_1, q8b.val1)
            var p3 = neon_sdot(SIMD[DType.int32, 4](0, 0, 0, 0), q2u_2, q8b.val2)
            var p4 = neon_sdot(SIMD[DType.int32, 4](0, 0, 0, 0), q2u_3, q8b.val3)

            # Pairwise add using NEON intrinsic (matching llama.cpp: vpaddq_s32(vpaddq_s32(p1, p2), vpaddq_s32(p3, p4)))
            var t1 = neon_vpaddq_s32(p1, p2)
            var t2 = neon_vpaddq_s32(p3, p4)
            var p = neon_vpaddq_s32(t1, t2)

            # Multiply-accumulate: sumi = sumi + p * scales32.val[ib64]
            var sc: SIMD[DType.int32, 4]
            if ib64 == 0:
                sc = scales32.val0
            elif ib64 == 1:
                sc = scales32.val1
            elif ib64 == 2:
                sc = scales32.val2
            else:
                sc = scales32.val3
            sumi = sumi + p * sc

        sumf += d * Float32(sumi[0] + sumi[1] + sumi[2] + sumi[3])

    return sumf * Float32(0.125)  # Divide by 8
