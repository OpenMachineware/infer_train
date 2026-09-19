# Q2_K × Q8_K optimized dot product
# Matches llama.cpp loop structure exactly

from std.memory import Pointer, unsafe_stack_allocation
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


@always_inline
def neon_smull(
    a: SIMD[DType.int16, 4],
    b: SIMD[DType.int16, 4],
) -> SIMD[DType.int32, 4]:
    return llvm_intrinsic[
        "llvm.aarch64.neon.smull.v4i32.v4i16",
        SIMD[DType.int32, 4],
        has_side_effect=False,
    ](a, b)


@always_inline
def vec_dot_q2_k_q8_k(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q2_K × Q8_K dot product - llama.cpp exact pattern."""
    var d = Float32(w_block.unsafe_offset(80).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())
    var dmin = Float32(w_block.unsafe_offset(82).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())
    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())

    var sc_ptr = w_block.unsafe_offset(0)
    var q2_ptr = w_block.unsafe_offset(16)
    var q8_ptr = q8_data.unsafe_offset(4)
    var q8_bsums = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()

    var m3b = SIMD[DType.uint8, 16](0x03)
    var m4b = SIMD[DType.uint8, 16](0x0F)
    var mzero = SIMD[DType.int32, 4](0)

    # Load scales+mins
    var scales_raw = sc_ptr.unsafe_load[width=16](offset=0)
    var scales_vec = scales_raw & m4b
    var mins_vec = scales_raw >> SIMD[DType.uint8, 16](4)

    # Store scales to stack (like llama.cpp's aux[16])
    var scales_aux = unsafe_stack_allocation[16, DType.uint8]()
    scales_aux.unsafe_store[width=16](offset=0, val=scales_vec)

    # Bias computation (same as llama.cpp)
    var mins_aux = unsafe_stack_allocation[16, DType.uint8]()
    mins_aux.unsafe_store[width=16](offset=0, val=mins_vec)

    var mins_ptr = mins_aux.unsafe_bitcast[Scalar[DType.int16]]()
    var mins_lo = mins_ptr.unsafe_load[width=4](offset=0)
    var mins_hi = mins_ptr.unsafe_load[width=4](offset=4)
    var mins_lo2 = mins_ptr.unsafe_load[width=4](offset=8)
    var mins_hi2 = mins_ptr.unsafe_load[width=4](offset=12)

    var bsums_ptr = q8_bsums
    var bsums_lo = bsums_ptr.unsafe_load[width=4](offset=0)
    var bsums_hi = bsums_ptr.unsafe_load[width=4](offset=4)
    var bsums_lo2 = bsums_ptr.unsafe_load[width=4](offset=8)
    var bsums_hi2 = bsums_ptr.unsafe_load[width=4](offset=12)

    var s0 = neon_smull(mins_lo, bsums_lo) + neon_smull(mins_hi, bsums_hi)
    var s1 = neon_smull(mins_lo2, bsums_lo2) + neon_smull(mins_hi2, bsums_hi2)
    var summs = neon_addv(s0 + s1)

    # Main loop (matching llama.cpp structure)
    var isum = Int32(0)
    var is = 0

    # Loop j=0 to 1 (QK_K/128 = 2)
    # j=0
    var q2bits = neon_ld1_u8_x2(q2_ptr)

    var q8bytes = neon_ld1_s8_x2(q8_ptr)
    var q2bytes_lo = (q2bits.lo & m3b).cast[DType.int8]()
    var q2bytes_hi = (q2bits.hi & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8bytes.lo)) * Int32(scales_aux.unsafe_offset(is).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8bytes.hi)) * Int32(scales_aux.unsafe_offset(is+1).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(32))
    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8bytes.lo)) * Int32(scales_aux.unsafe_offset(is+2).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8bytes.hi)) * Int32(scales_aux.unsafe_offset(is+3).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(64))
    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8bytes.lo)) * Int32(scales_aux.unsafe_offset(is+4).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8bytes.hi)) * Int32(scales_aux.unsafe_offset(is+5).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(96))
    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8bytes.lo)) * Int32(scales_aux.unsafe_offset(is+6).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8bytes.hi)) * Int32(scales_aux.unsafe_offset(is+7).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())

    is += 8

    # j=1
    q2bits = neon_ld1_u8_x2(q2_ptr.unsafe_offset(32))

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(128))
    q2bytes_lo = (q2bits.lo & m3b).cast[DType.int8]()
    q2bytes_hi = (q2bits.hi & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8bytes.lo)) * Int32(scales_aux.unsafe_offset(is).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8bytes.hi)) * Int32(scales_aux.unsafe_offset(is+1).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(160))
    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8bytes.lo)) * Int32(scales_aux.unsafe_offset(is+2).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8bytes.hi)) * Int32(scales_aux.unsafe_offset(is+3).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(192))
    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8bytes.lo)) * Int32(scales_aux.unsafe_offset(is+4).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8bytes.hi)) * Int32(scales_aux.unsafe_offset(is+5).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())

    q8bytes = neon_ld1_s8_x2(q8_ptr.unsafe_offset(224))
    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8bytes.lo)) * Int32(scales_aux.unsafe_offset(is+6).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8bytes.hi)) * Int32(scales_aux.unsafe_offset(is+7).unsafe_bitcast[Scalar[DType.uint8]]().unsafe_load())

    return d * q8_d * Float32(isum) - dmin * q8_d * Float32(summs)
