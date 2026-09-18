# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/simd/q6k_q8k_dot.mojo
#
# Q6_K × Q8_K int8 dot product using NEON SDOT instruction.
#
# Key optimization from Q4_K/Q5_K:
# 1. Pre-compute bias using vectorized operations
# 2. Use neon_ld1_u8_x2 for efficient loading
# 3. Use neon_sdot for dot product
# 4. Use neon_addv for horizontal sum

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic

comptime QK_K = 256

# NEON intrinsics (same as q4k_q8k_dot.mojo)

struct NeonU8x2(TrivialRegisterPassable):
    var lo: SIMD[DType.uint8, 16]
    var hi: SIMD[DType.uint8, 16]


@always_inline
def neon_ld1_u8_x2(ptr: Pointer[UInt8, MutUntrackedOrigin]) -> NeonU8x2:
    """Load 32 bytes using ld1.16b instruction."""
    return llvm_intrinsic[
        "llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonU8x2, has_side_effect=True
    ](ptr)


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
    """Horizontal sum using addv.4s - LLVM intrinsic."""
    return llvm_intrinsic[
        "llvm.vector.reduce.add.v4i32",
        Int32,
        has_side_effect=False,
    ](v)


@always_inline
def vec_dot_q6_k_q8_k(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q6_K × Q8_K dot product optimized with Q4_K/Q5_K pattern.

    Q6_K block layout (210 bytes):
    - ql: 128 bytes at offset 0 (lower 4 bits, 2 per byte)
    - qh: 64 bytes at offset 128 (upper 2 bits, 4 per byte)
    - scales: 16 bytes at offset 192 (int8 scales, 16 total)
    - d: fp16 scale at offset 208

    Q6_K value: 6-bit = (low4 | (high2 << 4)) - 32, range -32 to 31
    NO dmin term (bias from -32 offset)
    """
    var ql = w_block.unsafe_offset(0)
    var qh = w_block.unsafe_offset(128)
    var scales_ptr = w_block.unsafe_offset(192)
    var d = Float32(w_block.unsafe_offset(208).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())

    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())
    var q8_qs = q8_data.unsafe_offset(4).unsafe_bitcast[Scalar[DType.int8]]()
    var q8_bsums = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()

    # Pre-compute bias from -32 offset and bsums
    # bias = -32 * sum(scale[j] * bsum[j])
    # Use vectorized computation like Q4_K
    var scales_int8 = scales_ptr.unsafe_bitcast[Scalar[DType.int8]]()
    var scales_vec = scales_int8.unsafe_load[width=16](offset=0)

    var bsum0 = Int32(q8_bsums.unsafe_offset(0).unsafe_load())
    var bsum1 = Int32(q8_bsums.unsafe_offset(1).unsafe_load())
    var bsum2 = Int32(q8_bsums.unsafe_offset(2).unsafe_load())
    var bsum3 = Int32(q8_bsums.unsafe_offset(3).unsafe_load())
    var bsum4 = Int32(q8_bsums.unsafe_offset(4).unsafe_load())
    var bsum5 = Int32(q8_bsums.unsafe_offset(5).unsafe_load())
    var bsum6 = Int32(q8_bsums.unsafe_offset(6).unsafe_load())
    var bsum7 = Int32(q8_bsums.unsafe_offset(7).unsafe_load())
    var bsum8 = Int32(q8_bsums.unsafe_offset(8).unsafe_load())
    var bsum9 = Int32(q8_bsums.unsafe_offset(9).unsafe_load())
    var bsum10 = Int32(q8_bsums.unsafe_offset(10).unsafe_load())
    var bsum11 = Int32(q8_bsums.unsafe_offset(11).unsafe_load())
    var bsum12 = Int32(q8_bsums.unsafe_offset(12).unsafe_load())
    var bsum13 = Int32(q8_bsums.unsafe_offset(13).unsafe_load())
    var bsum14 = Int32(q8_bsums.unsafe_offset(14).unsafe_load())
    var bsum15 = Int32(q8_bsums.unsafe_offset(15).unsafe_load())

    var bias = Int32(scales_vec[0]) * bsum0 + Int32(scales_vec[1]) * bsum1 + \
               Int32(scales_vec[2]) * bsum2 + Int32(scales_vec[3]) * bsum3 + \
               Int32(scales_vec[4]) * bsum4 + Int32(scales_vec[5]) * bsum5 + \
               Int32(scales_vec[6]) * bsum6 + Int32(scales_vec[7]) * bsum7 + \
               Int32(scales_vec[8]) * bsum8 + Int32(scales_vec[9]) * bsum9 + \
               Int32(scales_vec[10]) * bsum10 + Int32(scales_vec[11]) * bsum11 + \
               Int32(scales_vec[12]) * bsum12 + Int32(scales_vec[13]) * bsum13 + \
               Int32(scales_vec[14]) * bsum14 + Int32(scales_vec[15]) * bsum15

    var m4b = SIMD[DType.uint8, 16](0x0F)
    var m2b = SIMD[DType.uint8, 16](3)
    var shift4 = SIMD[DType.uint8, 16](4)
    var shift2 = SIMD[DType.uint8, 16](2)

    # Load all Q6_K data upfront
    var ql_0 = ql.unsafe_load[width=16](offset=0)
    var ql_1 = ql.unsafe_load[width=16](offset=16)
    var ql_2 = ql.unsafe_load[width=16](offset=32)
    var ql_3 = ql.unsafe_load[width=16](offset=48)
    var ql_4 = ql.unsafe_load[width=16](offset=64)
    var ql_5 = ql.unsafe_load[width=16](offset=80)
    var ql_6 = ql.unsafe_load[width=16](offset=96)
    var ql_7 = ql.unsafe_load[width=16](offset=112)

    var qh_0 = qh.unsafe_load[width=16](offset=0)
    var qh_1 = qh.unsafe_load[width=16](offset=16)
    var qh_2 = qh.unsafe_load[width=16](offset=32)
    var qh_3 = qh.unsafe_load[width=16](offset=48)

    var mzero = SIMD[DType.int32, 4](0)
    var sumi = Int32(0)

    # j=0: ql bytes 0-63, qh bytes 0-31
    # Q6 values: (low4 | (high2 << 4)), range 0-63
    var q6_0 = ((ql_0 & m4b) | ((qh_0 & m2b) << shift4)).cast[DType.int8]()
    var q6_1 = ((ql_1 & m4b) | ((qh_1 & m2b) << shift4)).cast[DType.int8]()
    var q6_2 = ((ql_2 & m4b) | ((qh_0 & SIMD[DType.uint8, 16](12)) << shift2)).cast[DType.int8]()
    var q6_3 = ((ql_3 & m4b) | ((qh_1 & SIMD[DType.uint8, 16](12)) << shift2)).cast[DType.int8]()
    var q6_4 = ((ql_0 >> shift4) | (qh_0 & SIMD[DType.uint8, 16](48))).cast[DType.int8]()
    var q6_5 = ((ql_1 >> shift4) | (qh_1 & SIMD[DType.uint8, 16](48))).cast[DType.int8]()
    var q6_6 = ((ql_2 >> shift4) | ((qh_0 & SIMD[DType.uint8, 16](192)) >> shift2)).cast[DType.int8]()
    var q6_7 = ((ql_3 >> shift4) | ((qh_1 & SIMD[DType.uint8, 16](192)) >> shift2)).cast[DType.int8]()

    # Load Q8 values
    var q8_0 = q8_qs.unsafe_load[width=16](offset=0)
    var q8_1 = q8_qs.unsafe_load[width=16](offset=16)
    var q8_2 = q8_qs.unsafe_load[width=16](offset=32)
    var q8_3 = q8_qs.unsafe_load[width=16](offset=48)
    var q8_4 = q8_qs.unsafe_load[width=16](offset=64)
    var q8_5 = q8_qs.unsafe_load[width=16](offset=80)
    var q8_6 = q8_qs.unsafe_load[width=16](offset=96)
    var q8_7 = q8_qs.unsafe_load[width=16](offset=112)

    sumi += Int32(scales_vec[0]) * neon_addv(neon_sdot(mzero, q6_0, q8_0))
    sumi += Int32(scales_vec[1]) * neon_addv(neon_sdot(mzero, q6_1, q8_1))
    sumi += Int32(scales_vec[2]) * neon_addv(neon_sdot(mzero, q6_2, q8_2))
    sumi += Int32(scales_vec[3]) * neon_addv(neon_sdot(mzero, q6_3, q8_3))
    sumi += Int32(scales_vec[4]) * neon_addv(neon_sdot(mzero, q6_4, q8_4))
    sumi += Int32(scales_vec[5]) * neon_addv(neon_sdot(mzero, q6_5, q8_5))
    sumi += Int32(scales_vec[6]) * neon_addv(neon_sdot(mzero, q6_6, q8_6))
    sumi += Int32(scales_vec[7]) * neon_addv(neon_sdot(mzero, q6_7, q8_7))

    # j=1: ql bytes 64-127, qh bytes 32-63
    q6_0 = ((ql_4 & m4b) | ((qh_2 & m2b) << shift4)).cast[DType.int8]()
    q6_1 = ((ql_5 & m4b) | ((qh_3 & m2b) << shift4)).cast[DType.int8]()
    q6_2 = ((ql_6 & m4b) | ((qh_2 & SIMD[DType.uint8, 16](12)) << shift2)).cast[DType.int8]()
    q6_3 = ((ql_7 & m4b) | ((qh_3 & SIMD[DType.uint8, 16](12)) << shift2)).cast[DType.int8]()
    q6_4 = ((ql_4 >> shift4) | (qh_2 & SIMD[DType.uint8, 16](48))).cast[DType.int8]()
    q6_5 = ((ql_5 >> shift4) | (qh_3 & SIMD[DType.uint8, 16](48))).cast[DType.int8]()
    q6_6 = ((ql_6 >> shift4) | ((qh_2 & SIMD[DType.uint8, 16](192)) >> shift2)).cast[DType.int8]()
    q6_7 = ((ql_7 >> shift4) | ((qh_3 & SIMD[DType.uint8, 16](192)) >> shift2)).cast[DType.int8]()

    q8_0 = q8_qs.unsafe_load[width=16](offset=128)
    q8_1 = q8_qs.unsafe_load[width=16](offset=144)
    q8_2 = q8_qs.unsafe_load[width=16](offset=160)
    q8_3 = q8_qs.unsafe_load[width=16](offset=176)
    q8_4 = q8_qs.unsafe_load[width=16](offset=192)
    q8_5 = q8_qs.unsafe_load[width=16](offset=208)
    q8_6 = q8_qs.unsafe_load[width=16](offset=224)
    q8_7 = q8_qs.unsafe_load[width=16](offset=240)

    sumi += Int32(scales_vec[8]) * neon_addv(neon_sdot(mzero, q6_0, q8_0))
    sumi += Int32(scales_vec[9]) * neon_addv(neon_sdot(mzero, q6_1, q8_1))
    sumi += Int32(scales_vec[10]) * neon_addv(neon_sdot(mzero, q6_2, q8_2))
    sumi += Int32(scales_vec[11]) * neon_addv(neon_sdot(mzero, q6_3, q8_3))
    sumi += Int32(scales_vec[12]) * neon_addv(neon_sdot(mzero, q6_4, q8_4))
    sumi += Int32(scales_vec[13]) * neon_addv(neon_sdot(mzero, q6_5, q8_5))
    sumi += Int32(scales_vec[14]) * neon_addv(neon_sdot(mzero, q6_6, q8_6))
    sumi += Int32(scales_vec[15]) * neon_addv(neon_sdot(mzero, q6_7, q8_7))

    # Q6_K has a -32 bias
    return d * q8_d * (Float32(sumi) - 32.0 * Float32(bias))
