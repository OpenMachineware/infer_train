# Q2_K × Q8_K optimized dot product
# Uses vector widening for scales (like llama.cpp ARM NEON)
# NO scalar scale extraction - everything stays in vector registers

from std.memory import Pointer, unsafe_stack_allocation
from std.origin import MutUntrackedOrigin
from std.sys import llvm_intrinsic, inlined_assembly

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


# Widen uint8 to uint16 (ushll.8h)
@always_inline
def neon_ushll_u8(a: SIMD[DType.uint8, 8]) -> SIMD[DType.uint16, 8]:
    """Widen uint8x8 to uint16x8 (ushll.8h with shift=0)"""
    return inlined_assembly[
        "ushll $0.8h, $1.8b, #0",
        SIMD[DType.uint16, 8],
        SIMD[DType.uint8, 8],
        constraints="=w,w",
        has_side_effect=False,
    ](a)


# Widen uint16 to uint32 (ushll.4s)
@always_inline
def neon_ushll_u16(a: SIMD[DType.uint16, 4]) -> SIMD[DType.uint32, 4]:
    """Widen uint16x4 to uint32x4 (ushll.4s with shift=0)"""
    return inlined_assembly[
        "ushll $0.4s, $1.4h, #0",
        SIMD[DType.uint32, 4],
        SIMD[DType.uint16, 4],
        constraints="=w,w",
        has_side_effect=False,
    ](a)


# Widen uint16 high half to uint32 (ushll2.4s)
@always_inline
def neon_ushll2_u16(a: SIMD[DType.uint16, 8]) -> SIMD[DType.uint32, 4]:
    """Widen high half of uint16x8 to uint32x4 (ushll2.4s with shift=0)"""
    return inlined_assembly[
        "ushll2 $0.4s, $1.8h, #0",
        SIMD[DType.uint32, 4],
        SIMD[DType.uint16, 8],
        constraints="=w,w",
        has_side_effect=False,
    ](a)


# Insert scalar into vector lane (ins instruction)
# This is efficient - does NOT generate umov
@always_inline
def neon_vset_lane_s32[lane: Int](s: Int32, vec: SIMD[DType.int32, 4]) -> SIMD[DType.int32, 4]:
    """Insert scalar into vector lane - generates 'ins' instruction."""
    var result = vec
    result[lane] = s
    return result


@always_inline
def vec_dot_q2_k_q8_k(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q2_K × Q8_K dot product - vector widening for scales (llama.cpp pattern)."""
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

    # Widen scales to uint32 vectors (like llama.cpp)
    # scales_lo = scales[0:8], scales_hi = scales[8:16]
    var scales_lo_u8 = neon_vget_low_u8(scales_vec)
    var scales_hi_u8 = neon_vget_high_u8(scales_vec)

    # Widen to uint16
    var scales_lo_u16 = neon_ushll_u8(scales_lo_u8)
    var scales_hi_u16 = neon_ushll_u8(scales_hi_u8)

    # Widen to uint32 (split into 4-element chunks for dot products)
    var scales_0_3 = neon_ushll_u16(neon_vget_low_u16(scales_lo_u16))
    var scales_4_7 = neon_ushll_u16(neon_vget_high_u16(scales_lo_u16))
    var scales_8_11 = neon_ushll_u16(neon_vget_low_u16(scales_hi_u16))
    var scales_12_15 = neon_ushll_u16(neon_vget_high_u16(scales_hi_u16))

    # Bias: use widen multiply (vmull_s16) like llama.cpp
    # Store mins to stack and load via pointers (avoids umov)
    var mins_aux = unsafe_stack_allocation[16, DType.uint8]()
    mins_aux.unsafe_store[width=16](offset=0, val=mins_vec)

    # Split int16x8 into int16x4 pairs using pointer loads
    var mins_ptr = mins_aux.unsafe_bitcast[Scalar[DType.int16]]()
    var mins_lo_0_3 = mins_ptr.unsafe_load[width=4](offset=0)
    var mins_lo_4_7 = mins_ptr.unsafe_load[width=4](offset=4)
    var mins_hi_0_3 = mins_ptr.unsafe_load[width=4](offset=8)
    var mins_hi_4_7 = mins_ptr.unsafe_load[width=4](offset=12)

    var bsums_ptr = q8_bsums
    var bsums_lo_0_3 = bsums_ptr.unsafe_load[width=4](offset=0)
    var bsums_lo_4_7 = bsums_ptr.unsafe_load[width=4](offset=4)
    var bsums_hi_0_3 = bsums_ptr.unsafe_load[width=4](offset=8)
    var bsums_hi_4_7 = bsums_ptr.unsafe_load[width=4](offset=12)

    var s0 = neon_smull(mins_lo_0_3, bsums_lo_0_3) + neon_smull(mins_lo_4_7, bsums_lo_4_7)
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

    # Compute dot products and multiply by vector scales
    # j=0: q2 bytes 0-31
    var q2bits = neon_ld1_u8_x2(q2_ptr)

    var q2bytes_lo = (q2bits.lo & m3b).cast[DType.int8]()
    var q2bytes_hi = (q2bits.hi & m3b).cast[DType.int8]()

    # Compute 4 dot products, pack into vector, multiply by scales
    var dot_lo = SIMD[DType.int32, 4](0)
    dot_lo[0] = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_0.lo))
    dot_lo[1] = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_0.hi))

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    dot_lo[2] = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_1.lo))
    dot_lo[3] = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_1.hi))

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()

    var dot_hi = SIMD[DType.int32, 4](0)
    dot_hi[0] = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_2.lo))
    dot_hi[1] = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_2.hi))

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    dot_hi[2] = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_3.lo))
    dot_hi[3] = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_3.hi))

    # j=1: q2 bytes 32-63
    q2bits = neon_ld1_u8_x2(q2_ptr.unsafe_offset(32))

    q2bytes_lo = (q2bits.lo & m3b).cast[DType.int8]()
    q2bytes_hi = (q2bits.hi & m3b).cast[DType.int8]()

    var dot_lo2 = SIMD[DType.int32, 4](0)
    dot_lo2[0] = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_4.lo))
    dot_lo2[1] = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_4.hi))

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    dot_lo2[2] = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_5.lo))
    dot_lo2[3] = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_5.hi))

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()

    var dot_hi2 = SIMD[DType.int32, 4](0)
    dot_hi2[0] = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_6.lo))
    dot_hi2[1] = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_6.hi))

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    dot_hi2[2] = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_7.lo))
    dot_hi2[3] = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_7.hi))

    # Vector multiply by scales
    var result_lo = dot_lo * scales_0_3 + dot_hi * scales_4_7
    var result_hi = dot_lo2 * scales_8_11 + dot_hi2 * scales_12_15

    # Sum all results
    var isum_vec = result_lo + result_hi
    var isum = isum_vec[0] + isum_vec[1] + isum_vec[2] + isum_vec[3]

    return d * q8_d * Float32(isum) - dmin * q8_d * Float32(summs)
