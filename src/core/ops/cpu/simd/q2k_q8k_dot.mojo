# Q2_K × Q8_K optimized dot product
# Follows Q4_K pattern: pointer-based scalar loads for scales (NO SIMD element extracts)
# Uses ld1.16b, sdot, addv like llama.cpp ARM NEON

from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic

struct NeonU8x2(TrivialRegisterPassable):
    var lo: SIMD[DType.uint8, 16]
    var hi: SIMD[DType.uint8, 16]

struct NeonS8x2(TrivialRegisterPassable):
    var lo: SIMD[DType.int8, 16]
    var hi: SIMD[DType.int8, 16]


@always_inline
def neon_ld1_u8_x2(ptr: Pointer[UInt8, MutUntrackedOrigin]) -> NeonU8x2:
    return llvm_intrinsic[
        "llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonU8x2, has_side_effect=True
    ](ptr)


@always_inline
def neon_ld1_s8_x2(ptr: Pointer[UInt8, MutUntrackedOrigin]) -> NeonS8x2:
    return llvm_intrinsic[
        "llvm.aarch64.neon.ld1x2.v16i8.p0i8", NeonS8x2, has_side_effect=True
    ](ptr)


@always_inline
def neon_sdot(
    acc: SIMD[DType.int32, 4],
    a: SIMD[DType.int8, 16],
    b: SIMD[DType.int8, 16],
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.aarch64.neon.sdot.v4i32.v16i8",
        SIMD[DType.int32, 4],
        has_side_effect=False,
    ](acc, a, b)


@always_inline
def neon_addv(v: SIMD[DType.int32, 4]) -> Int32:
    return llvm_intrinsic[
        "llvm.vector.reduce.add.v4i32",
        Int32,
        has_side_effect=False,
    ](v)


# Widen multiply: int16 × int16 → int32 (vmull_s16)
@always_inline
def neon_smull(
    a: SIMD[DType.int16, 4],
    b: SIMD[DType.int16, 4],
) -> SIMD[DType.int32, 4]:
    """Signed multiply long: int16x4 × int16x4 → int32x4"""
    return llvm_intrinsic[
        "llvm.aarch64.neon.smull.v4i32.v4i16",
        SIMD[DType.int32, 4],
        has_side_effect=False,
    ](a, b)


@always_inline
def neon_addq_s32(
    a: SIMD[DType.int32, 4],
    b: SIMD[DType.int32, 4],
) -> SIMD[DType.int32, 4]:
    """Vector add int32x4"""
    return a + b  # Mojo handles this efficiently


@always_inline
def vec_dot_q2_k_q8_k(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q2_K × Q8_K dot product - follows Q4_K pattern exactly.

    Key: scales extracted via pointer byte loads (NOT SIMD element extracts).
    This avoids umov instructions that slow down SIMD element indexing.
    """
    var d = Float32(w_block.unsafe_offset(80).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())
    var dmin = Float32(w_block.unsafe_offset(82).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())
    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())

    var sc_ptr = w_block.unsafe_offset(0)
    var q2_ptr = w_block.unsafe_offset(16)
    var q8_ptr = q8_data.unsafe_offset(4)
    var q8_bsums = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()

    var m3b = SIMD[DType.uint8, 16](0x03)
    var mzero = SIMD[DType.int32, 4](0)

    # Load scales+mins as SIMD vector
    var scales_raw = sc_ptr.unsafe_load[width=16](offset=0)
    var m4b = SIMD[DType.uint8, 16](0x0F)
    var scales_vec = scales_raw & m4b
    var mins_vec = scales_raw >> SIMD[DType.uint8, 16](4)

    # Extract scales from SIMD (combine lo/hi dot products first)
    var sc0 = Int32(scales_vec[0])
    var sc1 = Int32(scales_vec[1])
    var sc2 = Int32(scales_vec[2])
    var sc3 = Int32(scales_vec[3])
    var sc4 = Int32(scales_vec[4])
    var sc5 = Int32(scales_vec[5])
    var sc6 = Int32(scales_vec[6])
    var sc7 = Int32(scales_vec[7])
    var sc8 = Int32(scales_vec[8])
    var sc9 = Int32(scales_vec[9])
    var sc10 = Int32(scales_vec[10])
    var sc11 = Int32(scales_vec[11])
    var sc12 = Int32(scales_vec[12])
    var sc13 = Int32(scales_vec[13])
    var sc14 = Int32(scales_vec[14])
    var sc15 = Int32(scales_vec[15])

    # Bias: use widen multiply (vmull_s16) like llama.cpp
    # Build int16x8 vectors, split into int16x4 pairs, widen multiply, then reduce
    var mins_lo = SIMD[DType.int16, 8](
        Int16(mins_vec[0]), Int16(mins_vec[1]), Int16(mins_vec[2]), Int16(mins_vec[3]),
        Int16(mins_vec[4]), Int16(mins_vec[5]), Int16(mins_vec[6]), Int16(mins_vec[7])
    )
    var mins_hi = SIMD[DType.int16, 8](
        Int16(mins_vec[8]), Int16(mins_vec[9]), Int16(mins_vec[10]), Int16(mins_vec[11]),
        Int16(mins_vec[12]), Int16(mins_vec[13]), Int16(mins_vec[14]), Int16(mins_vec[15])
    )
    var bsums_lo = q8_bsums.unsafe_load[width=8](offset=0)
    var bsums_hi = q8_bsums.unsafe_load[width=8](offset=8)

    # Split int16x8 into int16x4 pairs and use widen multiply (vmull_s16)
    # mins_lo/bsums_lo -> two smull results, add, then addv
    var mins_lo_0_3 = SIMD[DType.int16, 4](mins_lo[0], mins_lo[1], mins_lo[2], mins_lo[3])
    var mins_lo_4_7 = SIMD[DType.int16, 4](mins_lo[4], mins_lo[5], mins_lo[6], mins_lo[7])
    var bsums_lo_0_3 = SIMD[DType.int16, 4](bsums_lo[0], bsums_lo[1], bsums_lo[2], bsums_lo[3])
    var bsums_lo_4_7 = SIMD[DType.int16, 4](bsums_lo[4], bsums_lo[5], bsums_lo[6], bsums_lo[7])
    var s0 = neon_smull(mins_lo_0_3, bsums_lo_0_3) + neon_smull(mins_lo_4_7, bsums_lo_4_7)

    var mins_hi_0_3 = SIMD[DType.int16, 4](mins_hi[0], mins_hi[1], mins_hi[2], mins_hi[3])
    var mins_hi_4_7 = SIMD[DType.int16, 4](mins_hi[4], mins_hi[5], mins_hi[6], mins_hi[7])
    var bsums_hi_0_3 = SIMD[DType.int16, 4](bsums_hi[0], bsums_hi[1], bsums_hi[2], bsums_hi[3])
    var bsums_hi_4_7 = SIMD[DType.int16, 4](bsums_hi[4], bsums_hi[5], bsums_hi[6], bsums_hi[7])
    var s1 = neon_smull(mins_hi_0_3, bsums_hi_0_3) + neon_smull(mins_hi_4_7, bsums_hi_4_7)

    var summs = neon_addv(s0 + s1)

    # Preload all q8 data to reduce memory access latency
    var q8_0 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(0))
    var q8_1 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(32))
    var q8_2 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(64))
    var q8_3 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(96))
    var q8_4 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(128))
    var q8_5 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(160))
    var q8_6 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(192))
    var q8_7 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(224))

    # Compute all dot products first, then multiply by scales (improve ILP)
    var isum = Int32(0)

    # j=0: q2 bytes 0-31
    var q2bits = neon_ld1_u8_x2(q2_ptr)

    var q2bytes_lo = (q2bits.lo & m3b).cast[DType.int8]()
    var q2bytes_hi = (q2bits.hi & m3b).cast[DType.int8]()
    var dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_0.lo))
    var dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_0.hi))
    isum += dot0 * sc0 + dot1 * sc1

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_1.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_1.hi))
    isum += dot0 * sc2 + dot1 * sc3

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_2.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_2.hi))
    isum += dot0 * sc4 + dot1 * sc5

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_3.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_3.hi))
    isum += dot0 * sc6 + dot1 * sc7

    # j=1: q2 bytes 32-63
    q2bits = neon_ld1_u8_x2(q2_ptr.unsafe_offset(32))

    q2bytes_lo = (q2bits.lo & m3b).cast[DType.int8]()
    q2bytes_hi = (q2bits.hi & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_4.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_4.hi))
    isum += dot0 * sc8 + dot1 * sc9

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_5.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_5.hi))
    isum += dot0 * sc10 + dot1 * sc11

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_6.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_6.hi))
    isum += dot0 * sc12 + dot1 * sc13

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_7.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_7.hi))
    isum += dot0 * sc14 + dot1 * sc15

    return d * q8_d * Float32(isum) - dmin * q8_d * Float32(summs)
