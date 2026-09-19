# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Jia Liu & InferTrain contributors
# core/ops/cpu/simd/q5k_q8k_dot.mojo
#
# Q5_K × Q8_K int8 dot product using NEON SDOT instruction.
#
# Key optimization from Q4_K:
# 1. Pre-compute ALL scales at the beginning
# 2. Pre-compute ALL min values at the beginning
# 3. Compute bias in ONE operation
# 4. Use neon_ld1_u8_x2 for efficient loading
# 5. Use neon_sdot for dot product
# 6. Use neon_addv for horizontal sum

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
def neon_smull(
    a: SIMD[DType.int16, 4],
    b: SIMD[DType.int16, 4],
) -> SIMD[DType.int32, 4]:
    """NEON SMULL: int16 × int16 -> int32 widening multiply."""
    return llvm_intrinsic[
        "llvm.aarch64.neon.smull.v4i32",
        SIMD[DType.int32, 4],
        has_side_effect=False,
    ](a, b)


@always_inline
def vec_dot_q5_k_q8_k(
    w_block: Pointer[UInt8, MutUntrackedOrigin],
    q8_data: Pointer[UInt8, MutUntrackedOrigin],
) -> Float32:
    """Q5_K × Q8_K dot product optimized with Q4_K pattern.

    Q5_K block layout (176 bytes):
    - d: fp16 scale at offset 0
    - dmin: fp16 min scale at offset 2
    - scales: 12 bytes at offset 4 (packed 6-bit scales and mins)
    - qh: 32 bytes at offset 16 (high bits, 1 bit per element, packed)
    - qs: 128 bytes at offset 48 (low 4 bits, 256 elements packed)

    Q5_K value: 5-bit = low4 + (high_bit ? 16 : 0), range 0-31

    Q8_K layout (292 bytes):
    - d: float32 scale at offset 0
    - qs: 256 int8 at offset 4
    - bsums: 16 int16 at offset 260
    """
    var mzero = SIMD[DType.int32, 4](0)
    var m4b = SIMD[DType.uint8, 16](0x0F)

    # Load super-block scales
    var d = Float32(w_block.unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())
    var dmin = Float32(w_block.unsafe_offset(2).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load())
    var q8_d = Float32(q8_data.unsafe_bitcast[Scalar[DType.float32]]().unsafe_load())

    # Pre-compute scales using llama.cpp unpacking method
    # Q5_K scales: 12 bytes packed into utmp[4] (uint32 array)
    # Layout after unpack:
    #   utmp[0]: scales 0-3 (as uint8)
    #   utmp[1]: scales 4-7 (as uint8)
    #   utmp[2]: mins 0-3 (as uint8)
    #   utmp[3]: mins 4-7 (as uint8)

    # Load 12 bytes as 3 uint32
    var scales_ptr = w_block.unsafe_offset(4).unsafe_bitcast[UInt32]()
    var utmp0 = scales_ptr.unsafe_load(offset=0)
    var utmp1 = scales_ptr.unsafe_load(offset=1)
    var utmp2 = scales_ptr.unsafe_load(offset=2)

    # Unpack per llama.cpp algorithm
    # utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0f) | (((utmp[1] >> 6) & 0x03030303) << 4)
    var utmp3 = ((utmp2 >> 4) & 0x0F0F0F0F) | (((utmp1 >> 6) & 0x03030303) << 4)

    # const uint32_t uaux = utmp[1] & 0x3f3f3f3f
    var uaux = utmp1 & 0x3F3F3F3F

    # utmp[1] = (utmp[2] & 0x0f0f0f0f) | (((utmp[0] >> 6) & 0x03030303) << 4)
    utmp1 = (utmp2 & 0x0F0F0F0F) | (((utmp0 >> 6) & 0x03030303) << 4)

    # utmp[2] = uaux
    utmp2 = uaux

    # utmp[0] &= 0x3f3f3f3f
    utmp0 = utmp0 & 0x3F3F3F3F

    # Now extract scales as uint8 (16 bytes total)

    # Pre-compute bias from Q8_K bsums using SIMD widening multiply
    # Load bsums as 16 int16 values
    var bsums_ptr = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
    var bsums = SIMD[DType.int16, 16](
        bsums_ptr.unsafe_load(offset=0),
        bsums_ptr.unsafe_load(offset=1),
        bsums_ptr.unsafe_load(offset=2),
        bsums_ptr.unsafe_load(offset=3),
        bsums_ptr.unsafe_load(offset=4),
        bsums_ptr.unsafe_load(offset=5),
        bsums_ptr.unsafe_load(offset=6),
        bsums_ptr.unsafe_load(offset=7),
        bsums_ptr.unsafe_load(offset=8),
        bsums_ptr.unsafe_load(offset=9),
        bsums_ptr.unsafe_load(offset=10),
        bsums_ptr.unsafe_load(offset=11),
        bsums_ptr.unsafe_load(offset=12),
        bsums_ptr.unsafe_load(offset=13),
        bsums_ptr.unsafe_load(offset=14),
        bsums_ptr.unsafe_load(offset=15),
    )
    # Pairwise add: 16 int16 -> 8 int16
    # Use neon_addp for pairwise add
    var q8sums = SIMD[DType.int16, 8](
        bsums[0] + bsums[1],
        bsums[2] + bsums[3],
        bsums[4] + bsums[5],
        bsums[6] + bsums[7],
        bsums[8] + bsums[9],
        bsums[10] + bsums[11],
        bsums[12] + bsums[13],
        bsums[14] + bsums[15],
    )

    # Extract mins from utmp2 and utmp3 (as uint8)
    # utmp2 and utmp3 are uint32, each contains 4 uint8 mins
    var mins_bytes = SIMD[DType.uint8, 8](
        UInt8(utmp2 & 0xFF),
        UInt8((utmp2 >> 8) & 0xFF),
        UInt8((utmp2 >> 16) & 0xFF),
        UInt8((utmp2 >> 24) & 0xFF),
        UInt8(utmp3 & 0xFF),
        UInt8((utmp3 >> 8) & 0xFF),
        UInt8((utmp3 >> 16) & 0xFF),
        UInt8((utmp3 >> 24) & 0xFF),
    )

    # Widening multiply: int16 * uint8 -> int32
    # First widen uint8 to int16
    var mins_wide = mins_bytes.cast[DType.int16]()

    # Split q8sums into low and high halves
    var q8sums_lo = SIMD[DType.int16, 4](
        q8sums[0], q8sums[1], q8sums[2], q8sums[3]
    )
    var q8sums_hi = SIMD[DType.int16, 4](
        q8sums[4], q8sums[5], q8sums[6], q8sums[7]
    )
    var mins_lo = SIMD[DType.int16, 4](
        mins_wide[0], mins_wide[1], mins_wide[2], mins_wide[3]
    )
    var mins_hi = SIMD[DType.int16, 4](
        mins_wide[4], mins_wide[5], mins_wide[6], mins_wide[7]
    )

    # Use neon_smull for widening multiply
    var prod_lo = neon_smull(q8sums_lo, mins_lo)
    var prod_hi = neon_smull(q8sums_hi, mins_hi)

    # Add and reduce
    var prod_sum = prod_lo + prod_hi
    var sumi_mins = neon_addv(prod_sum)

    var bias = dmin * q8_d * Float32(sumi_mins)

    # Extract scales as uint8 array from unpacked utmp0-1
    # utmp0: scales 0-3 (each uint32 contains 4 uint8)
    # utmp1: scales 4-7
    var scales = SIMD[DType.uint8, 8](
        UInt8(utmp0 & 0xFF),
        UInt8((utmp0 >> 8) & 0xFF),
        UInt8((utmp0 >> 16) & 0xFF),
        UInt8((utmp0 >> 24) & 0xFF),
        UInt8(utmp1 & 0xFF),
        UInt8((utmp1 >> 8) & 0xFF),
        UInt8((utmp1 >> 16) & 0xFF),
        UInt8((utmp1 >> 24) & 0xFF),
    )

    # Base pointers
    var qh = w_block.unsafe_offset(16)
    var qs = w_block.unsafe_offset(48)
    var q8_qs = q8_data.unsafe_offset(4)

    # Load qh bits once (32 bytes = 256 bits, one bit per element)
    var qhbits_0 = qh.unsafe_load[width=16](offset=0)
    var qhbits_1 = qh.unsafe_load[width=16](offset=16)

    var mone = SIMD[DType.uint8, 16](1)
    var mtwo = SIMD[DType.uint8, 16](2)
    var shift4 = SIMD[DType.uint8, 16](4)
    var shift3 = SIMD[DType.uint8, 16](3)

    var sumi = Int32(0)

    # j=0: bytes 0-31 of qs, qh bits 0
    var q5bits = neon_ld1_u8_x2(qs)
    var q8bytes = neon_ld1_u8_x2(q8_qs)

    # Extract high bits: bit 0 of each qh byte
    var q5h_0 = (qhbits_0 & mone) << shift4
    var q5h_1 = (qhbits_1 & mone) << shift4

    # Combine low nibble with high bit to get 5-bit value
    var q5_0 = ((q5bits.lo & m4b) | q5h_0).cast[DType.int8]()
    var q5_1 = ((q5bits.hi & m4b) | q5h_1).cast[DType.int8]()

    # Load Q8 for low nibble part
    var dot_0 = neon_sdot(neon_sdot(mzero, q5_0, q8bytes.lo.cast[DType.int8]()),
                           q5_1, q8bytes.hi.cast[DType.int8]())

    # High nibble: bit 1 of each qh byte, shifted by 3
    var q5h_2 = (qhbits_0 & mtwo) << shift3
    var q5h_3 = (qhbits_1 & mtwo) << shift3
    var q5_2 = ((q5bits.lo >> shift4) | q5h_2).cast[DType.int8]()
    var q5_3 = ((q5bits.hi >> shift4) | q5h_3).cast[DType.int8]()

    # Load Q8 for high nibble part
    var q8bytes_hi = neon_ld1_u8_x2(q8_qs.unsafe_offset(32))
    var dot_1 = neon_sdot(neon_sdot(mzero, q5_2, q8bytes_hi.lo.cast[DType.int8]()),
                           q5_3, q8bytes_hi.hi.cast[DType.int8]())

    sumi += neon_addv(dot_0) * Int32(scales[0]) + neon_addv(dot_1) * Int32(scales[1])

    # j=1: shift qh by 2 bits
    qhbits_0 = qhbits_0 >> SIMD[DType.uint8, 16](2)
    qhbits_1 = qhbits_1 >> SIMD[DType.uint8, 16](2)

    q5bits = neon_ld1_u8_x2(qs.unsafe_offset(32))
    q8bytes = neon_ld1_u8_x2(q8_qs.unsafe_offset(64))

    q5h_0 = (qhbits_0 & mone) << shift4
    q5h_1 = (qhbits_1 & mone) << shift4
    q5_0 = ((q5bits.lo & m4b) | q5h_0).cast[DType.int8]()
    q5_1 = ((q5bits.hi & m4b) | q5h_1).cast[DType.int8]()

    dot_0 = neon_sdot(neon_sdot(mzero, q5_0, q8bytes.lo.cast[DType.int8]()),
                      q5_1, q8bytes.hi.cast[DType.int8]())

    q5h_2 = (qhbits_0 & mtwo) << shift3
    q5h_3 = (qhbits_1 & mtwo) << shift3
    q5_2 = ((q5bits.lo >> shift4) | q5h_2).cast[DType.int8]()
    q5_3 = ((q5bits.hi >> shift4) | q5h_3).cast[DType.int8]()

    q8bytes_hi = neon_ld1_u8_x2(q8_qs.unsafe_offset(96))
    dot_1 = neon_sdot(neon_sdot(mzero, q5_2, q8bytes_hi.lo.cast[DType.int8]()),
                      q5_3, q8bytes_hi.hi.cast[DType.int8]())

    sumi += neon_addv(dot_0) * Int32(scales[2]) + neon_addv(dot_1) * Int32(scales[3])

    # j=2: shift qh by 4 bits (total)
    qhbits_0 = qhbits_0 >> SIMD[DType.uint8, 16](2)
    qhbits_1 = qhbits_1 >> SIMD[DType.uint8, 16](2)

    q5bits = neon_ld1_u8_x2(qs.unsafe_offset(64))
    q8bytes = neon_ld1_u8_x2(q8_qs.unsafe_offset(128))

    q5h_0 = (qhbits_0 & mone) << shift4
    q5h_1 = (qhbits_1 & mone) << shift4
    q5_0 = ((q5bits.lo & m4b) | q5h_0).cast[DType.int8]()
    q5_1 = ((q5bits.hi & m4b) | q5h_1).cast[DType.int8]()

    dot_0 = neon_sdot(neon_sdot(mzero, q5_0, q8bytes.lo.cast[DType.int8]()),
                      q5_1, q8bytes.hi.cast[DType.int8]())

    q5h_2 = (qhbits_0 & mtwo) << shift3
    q5h_3 = (qhbits_1 & mtwo) << shift3
    q5_2 = ((q5bits.lo >> shift4) | q5h_2).cast[DType.int8]()
    q5_3 = ((q5bits.hi >> shift4) | q5h_3).cast[DType.int8]()

    q8bytes_hi = neon_ld1_u8_x2(q8_qs.unsafe_offset(160))
    dot_1 = neon_sdot(neon_sdot(mzero, q5_2, q8bytes_hi.lo.cast[DType.int8]()),
                      q5_3, q8bytes_hi.hi.cast[DType.int8]())

    sumi += neon_addv(dot_0) * Int32(scales[4]) + neon_addv(dot_1) * Int32(scales[5])

    # j=3: shift qh by 6 bits (total)
    qhbits_0 = qhbits_0 >> SIMD[DType.uint8, 16](2)
    qhbits_1 = qhbits_1 >> SIMD[DType.uint8, 16](2)

    q5bits = neon_ld1_u8_x2(qs.unsafe_offset(96))
    q8bytes = neon_ld1_u8_x2(q8_qs.unsafe_offset(192))

    q5h_0 = (qhbits_0 & mone) << shift4
    q5h_1 = (qhbits_1 & mone) << shift4
    q5_0 = ((q5bits.lo & m4b) | q5h_0).cast[DType.int8]()
    q5_1 = ((q5bits.hi & m4b) | q5h_1).cast[DType.int8]()

    dot_0 = neon_sdot(neon_sdot(mzero, q5_0, q8bytes.lo.cast[DType.int8]()),
                      q5_1, q8bytes.hi.cast[DType.int8]())

    q5h_2 = (qhbits_0 & mtwo) << shift3
    q5h_3 = (qhbits_1 & mtwo) << shift3
    q5_2 = ((q5bits.lo >> shift4) | q5h_2).cast[DType.int8]()
    q5_3 = ((q5bits.hi >> shift4) | q5h_3).cast[DType.int8]()

    q8bytes_hi = neon_ld1_u8_x2(q8_qs.unsafe_offset(224))
    dot_1 = neon_sdot(neon_sdot(mzero, q5_2, q8bytes_hi.lo.cast[DType.int8]()),
                      q5_3, q8bytes_hi.hi.cast[DType.int8]())

    sumi += neon_addv(dot_0) * Int32(scales[6]) + neon_addv(dot_1) * Int32(scales[7])

    # Apply super-block scale
    return d * q8_d * Float32(sumi) - bias
