# Q2_K × Q8_K optimized dot product
# Uses SIMD store + array access for scales (like llama.cpp)

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
def neon_vst1q_u8(ptr: Pointer[UInt8, MutUntrackedOrigin], value: SIMD[DType.uint8, 16]):
    """Store SIMD vector to memory"""
    ptr.unsafe_store[width=16](offset=0, val=value)


@always_inline
def vec_dot_q2_k_q8_k(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q2_K × Q8_K dot product - optimized with SIMD store for scales."""
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

    # Load scales+mins as SIMD vector and store to stack array (like llama.cpp)
    var scales_raw = sc_ptr.unsafe_load[width=16](offset=0)
    var scales_vec = scales_raw & m4b
    var mins_vec = scales_raw >> SIMD[DType.uint8, 16](4)

    # Store scales to stack array
    var scales_aux = Pointer[UInt8, MutUntrackedOrigin].stack_alloc[16]()
    neon_vst1q_u8(scales_aux, scales_vec)

    # Compute bias using SIMD (like llama.cpp)
    # Load bsums and mins as int16 vectors
    var bsums_lo = q8_bsums.unsafe_load[width=8](offset=0)
    var bsums_hi = q8_bsums.unsafe_load[width=8](offset=8)

    # Widen mins from uint8 to int16
    var mins_lo_u8 = SIMD[DType.uint8, 8](
        UInt8(mins_vec[0]), UInt8(mins_vec[1]), UInt8(mins_vec[2]), UInt8(mins_vec[3]),
        UInt8(mins_vec[4]), UInt8(mins_vec[5]), UInt8(mins_vec[6]), UInt8(mins_vec[7])
    )
    var mins_hi_u8 = SIMD[DType.uint8, 8](
        UInt8(mins_vec[8]), UInt8(mins_vec[9]), UInt8(mins_vec[10]), UInt8(mins_vec[11]),
        UInt8(mins_vec[12]), UInt8(mins_vec[13]), UInt8(mins_vec[14]), UInt8(mins_vec[15])
    )
    var mins_lo = mins_lo_u8.cast[DType.int16]()
    var mins_hi = mins_hi_u8.cast[DType.int16]()

    # Compute summs using widen multiply (vmull_s16)
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

    # Preload all q8 data
    var q8_0 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(0))
    var q8_1 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(32))
    var q8_2 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(64))
    var q8_3 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(96))
    var q8_4 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(128))
    var q8_5 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(160))
    var q8_6 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(192))
    var q8_7 = neon_ld1_s8_x2(q8_ptr.unsafe_offset(224))

    var isum = Int32(0)
    var scale_idx = 0

    # j=0: q2 bytes 0-31
    var q2bits = neon_ld1_u8_x2(q2_ptr)

    var q2bytes_lo = (q2bits.lo & m3b).cast[DType.int8]()
    var q2bytes_hi = (q2bits.hi & m3b).cast[DType.int8]()
    var dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_0.lo))
    var dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_0.hi))
    isum += dot0 * Int32(scales_aux[scale_idx]) + dot1 * Int32(scales_aux[scale_idx+1])

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_1.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_1.hi))
    isum += dot0 * Int32(scales_aux[scale_idx+2]) + dot1 * Int32(scales_aux[scale_idx+3])

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_2.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_2.hi))
    isum += dot0 * Int32(scales_aux[scale_idx+4]) + dot1 * Int32(scales_aux[scale_idx+5])

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_3.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_3.hi))
    isum += dot0 * Int32(scales_aux[scale_idx+6]) + dot1 * Int32(scales_aux[scale_idx+7])

    scale_idx += 8

    # j=1: q2 bytes 32-63
    q2bits = neon_ld1_u8_x2(q2_ptr.unsafe_offset(32))

    q2bytes_lo = (q2bits.lo & m3b).cast[DType.int8]()
    q2bytes_hi = (q2bits.hi & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_4.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_4.hi))
    isum += dot0 * Int32(scales_aux[scale_idx]) + dot1 * Int32(scales_aux[scale_idx+1])

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](2)) & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_5.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_5.hi))
    isum += dot0 * Int32(scales_aux[scale_idx+2]) + dot1 * Int32(scales_aux[scale_idx+3])

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](4)) & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_6.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_6.hi))
    isum += dot0 * Int32(scales_aux[scale_idx+4]) + dot1 * Int32(scales_aux[scale_idx+5])

    q2bytes_lo = ((q2bits.lo >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    q2bytes_hi = ((q2bits.hi >> SIMD[DType.uint8, 16](6)) & m3b).cast[DType.int8]()
    dot0 = neon_addv(neon_sdot(mzero, q2bytes_lo, q8_7.lo))
    dot1 = neon_addv(neon_sdot(mzero, q2bytes_hi, q8_7.hi))
    isum += dot0 * Int32(scales_aux[scale_idx+6]) + dot1 * Int32(scales_aux[scale_idx+7])

    return d * q8_d * Float32(isum) - dmin * q8_d * Float32(summs)
