# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# IQ4_XS × Q8_K SIMD-optimized kernel using NEON TBL and SDOT
#
# Key optimization pattern from llama.cpp:
# 1. ld1.16b for efficient loading
# 2. vqtbl1q_s8 (TBL) for parallel table lookup
# 3. sdot.4s for vector dot product
# 4. addv.4s for horizontal sum

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic

comptime QK_K = 256

# Non-linear quantization values for IQ4 formats
comptime kvalues_iq4nl = SIMD[DType.int8, 16](
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
)

# NEON intrinsics

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
    # Load as uint8 then reinterpret
    var ptr_s8 = ptr.unsafe_bitcast[Pointer[Int8, MutUntrackedOrigin]]()
    return llvm_intrinsic[
        "llvm.aarch64.neon.ld1x4.v16i8.p0i8", NeonS8x4, has_side_effect=True
    ](ptr_s8)


@always_inline
def neon_tbl1(table: SIMD[DType.int8, 16], indices: SIMD[DType.uint8, 16]) -> SIMD[DType.int8, 16]:
    """NEON TBL1: parallel table lookup for 16 values."""
    return llvm_intrinsic[
        "llvm.aarch64.neon.tbl1.v16i8",
        SIMD[DType.int8, 16],
        has_side_effect=False,
    ](table, indices)


@always_inline
def neon_sdot(
    acc: SIMD[DType.int32, 4],
    a: SIMD[DType.int8, 16],
    b: SIMD[DType.int8, 16],
) -> SIMD[DType.int32, 4]:
    """NEON SDOT: int8 × int8 -> int32 dot product."""
    return llvm_intrinsic[
        "llvm.aarch64.neon.sdot.v4i32.v16i8",
        SIMD[DType.int32, 4],
        has_side_effect=False,
    ](acc, a, b)


@always_inline
def neon_addv(v: SIMD[DType.int32, 4]) -> Int32:
    """Horizontal sum using addv.4s."""
    return llvm_intrinsic[
        "llvm.vector.reduce.add.v4i32",
        Int32,
        has_side_effect=False,
    ](v)


def vec_dot_iq4xs_q8k(
    x: Pointer[UInt8, MutUntrackedOrigin],
    y: Pointer[UInt8, MutUntrackedOrigin],
    nb: Int,
) -> Float32:
    """Compute dot product of IQ4_XS weight block × Q8_K activation block.

    x points to nb blocks of IQ4_XS (136 bytes each)
    y points to nb blocks of Q8_K (292 bytes each)

    Returns the dot product.
    """
    # Load the kvalues table once
    var values = kvalues_iq4nl
    var m4b = SIMD[DType.uint8, 16](0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
                                     0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f)

    var sumf = Float32(0)

    for ibl in range(nb):
        # Load super-block scale (FP16 at offset 0)
        var d_ptr = x.unsafe_offset(ibl * 136).unsafe_bitcast[Scalar[DType.float16]]()
        var d = Float32(d_ptr.unsafe_load[width=1](offset=0))

        # Load Q8_K scale (FP32 at offset 0)
        var y_d_ptr = y.unsafe_offset(ibl * 292).unsafe_bitcast[Scalar[DType.float32]]()
        var y_d = Float32(y_d_ptr.unsafe_load[width=1](offset=0))

        # Load scales_h (uint16 at offset 2)
        var scales_h_raw = x.unsafe_offset(ibl * 136 + 2)
        var h = UInt16(scales_h_raw.unsafe_load[width=1](offset=0)) |
                (UInt16(scales_h_raw.unsafe_load[width=1](offset=1)) << 8)

        var sumi1 = 0
        var sumi2 = 0

        # Process 4 pairs of sub-blocks (ib=0,2,4,6, each pair has 2 sub-blocks)
        # Total 8 sub-blocks of 32 elements = 256 elements
        for ib in range(0, 8, 2):
            # Load scales_l[ib/2]
            var scales_l = x.unsafe_load[width=1](offset=ibl * 136 + 4 + ib // 2)

            # Decode scales (use current h, then shift)
            var ls1 = Int(scales_l & 0xf) | Int((h << 4) & 0x30)
            ls1 = ls1 - 32
            var ls2 = Int(scales_l >> 4) | Int((h << 2) & 0x30)
            ls2 = ls2 - 32
            h = h >> 4

            # Process first sub-block in this pair (sub-block ib)
            var q4_ptr = x.unsafe_offset(ibl * 136 + 8 + ib * 16)
            var q4bits = neon_ld1_u8_x2(q4_ptr)

            var q8_ptr = y.unsafe_offset(ibl * 292 + 4 + ib * 32)
            var q8b = neon_ld1_s8_x4(q8_ptr)

            var q4b_val0 = neon_tbl1(values, q4bits.val0 & m4b)
            var q4b_val1 = neon_tbl1(values, q4bits.val0 >> 4)

            var prod_1 = neon_sdot(SIMD[DType.int32, 4](0), q4b_val0, q8b.val0)
            prod_1 = neon_sdot(prod_1, q4b_val1, q8b.val1)

            sumi1 += Int(neon_addv(prod_1)) * ls1

            # Process second sub-block in this pair (sub-block ib+1)
            var q4b_val2 = neon_tbl1(values, q4bits.val1 & m4b)
            var q4b_val3 = neon_tbl1(values, q4bits.val1 >> 4)

            var prod_2 = neon_sdot(SIMD[DType.int32, 4](0), q4b_val2, q8b.val2)
            prod_2 = neon_sdot(prod_2, q4b_val3, q8b.val3)

            sumi2 += Int(neon_addv(prod_2)) * ls2

        sumf += d * y_d * Float32(sumi1 + sumi2)

    return sumf
