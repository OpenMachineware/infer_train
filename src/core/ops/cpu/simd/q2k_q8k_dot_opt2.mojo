# Q2_K × Q8_K optimized dot product
# Hybrid approach: vector widening for scales, pointer-based extraction

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
    """Q2_K × Q8_K dot product - optimized hybrid approach."""
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

    # Store scales to stack for scalar access
    var scales_aux = unsafe_stack_allocation[16, DType.uint8]()
    scales_aux.unsafe_store[width=16](offset=0, val=scales_vec)

    # Extract scales via pointer loads (avoids umov)
    var scales_ptr = scales_aux.unsafe_bitcast[Scalar[DType.uint8]]()

    # Bias: use widen multiply (vmull_s16)
    var mins_aux = unsafe_stack_allocation[16, DType.uint8]()
    mins_aux.unsafe_store[width=16](offset=0, val=mins_vec)

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

    # Preload all q8 data
    var q8_0 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(0))
    var q8_1 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(32))
    var q8_2 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(64))
    var q8_3 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(96))
    var q8_4 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(128))
    var q8_5 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(160))
    var q8_6 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(192))
    var q8_7 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(224))

    # Compute dot products with immediate scale multiply
    var isum = Int32(0)

    # j=0: q2 bytes 0-31
    var q2bits = neon_ld1_u8_x2(q2_ptr)

    var q2bytes_lo = (q2bits.lo & m3b).cast[DType.int8]()
    var q2bytes_hi = (q2bits.hi & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8_0.lo)) * Int32(scales_ptr.unsafe_load[width=1](offset=0)[0])
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8_0.hi)) * Int32(scales_ptr.unsafe_load[width=1](offset=1)[0])

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8_1.lo)) * Int32(scales_ptr.unsafe_load[width=1](offset=2)[0])
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8_1.hi)) * Int32(scales_ptr.unsafe_load[width=1](offset=3)[0])

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8_2.lo)) * Int32(scales_ptr.unsafe_load[width=1](offset=4)[0])
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8_2.hi)) * Int32(scales_ptr.unsafe_load[width=1](offset=5)[0])

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8_3.lo)) * Int32(scales_ptr.unsafe_load[width=1](offset=6)[0])
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8_3.hi)) * Int32(scales_ptr.unsafe_load[width=1](offset=7)[0])

    # j=1: q2 bytes 32-63
    q2bits = neon_ld1_u8_x2(q2_ptr.unsafe_offset(32))

    q2bytes_lo = (q2bits.lo & m3b).cast[DType.int8]()
    q2bytes_hi = (q2bits.hi & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8_4.lo)) * Int32(scales_ptr.unsafe_load[width=1](offset=8)[0])
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8_4.hi)) * Int32(scales_ptr.unsafe_load[width=1](offset=9)[0])

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8_5.lo)) * Int32(scales_ptr.unsafe_load[width=1](offset=10)[0])
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8_5.hi)) * Int32(scales_ptr.unsafe_load[width=1](offset=11)[0])

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8_6.lo)) * Int32(scales_ptr.unsafe_load[width=1](offset=12)[0])
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8_6.hi)) * Int32(scales_ptr.unsafe_load[width=1](offset=13)[0])

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    isum += neon_addv(neon_sdot(mzero, q2bytes_lo, q8_7.lo)) * Int32(scales_ptr.unsafe_load[width=1](offset=14)[0])
    isum += neon_addv(neon_sdot(mzero, q2bytes_hi, q8_7.hi)) * Int32(scales_ptr.unsafe_load[width=1](offset=15)[0])

    return d * q8_d * Float32(isum) - dmin * q8_d * Float32(summs)
