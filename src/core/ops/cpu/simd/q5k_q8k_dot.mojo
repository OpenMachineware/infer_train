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

    # Pre-compute ALL scales at once (same pattern as Q4_K)
    # Q5_K scales are stored in the same layout as Q4_K: 12 bytes packed
    var scales_raw = w_block.unsafe_offset(4)

    # Scales 0-3: lower 6 bits of first 4 bytes
    var sc0 = Int(scales_raw.unsafe_load[width=1](offset=0).value()) & 0x3F
    var sc1 = Int(scales_raw.unsafe_load[width=1](offset=1).value()) & 0x3F
    var sc2 = Int(scales_raw.unsafe_load[width=1](offset=2).value()) & 0x3F
    var sc3 = Int(scales_raw.unsafe_load[width=1](offset=3).value()) & 0x3F

    # Scales 4-7: packed with upper bits
    var sc4 = (Int(scales_raw.unsafe_load[width=1](offset=8).value()) & 0x0F) | \
              ((Int(scales_raw.unsafe_load[width=1](offset=0).value()) >> 6) << 4)
    var sc5 = (Int(scales_raw.unsafe_load[width=1](offset=9).value()) & 0x0F) | \
              ((Int(scales_raw.unsafe_load[width=1](offset=1).value()) >> 6) << 4)
    var sc6 = (Int(scales_raw.unsafe_load[width=1](offset=10).value()) & 0x0F) | \
              ((Int(scales_raw.unsafe_load[width=1](offset=2).value()) >> 6) << 4)
    var sc7 = (Int(scales_raw.unsafe_load[width=1](offset=11).value()) & 0x0F) | \
              ((Int(scales_raw.unsafe_load[width=1](offset=3).value()) >> 6) << 4)

    # Pre-compute ALL min values at once
    # Mins 0-3: lower 6 bits of bytes 4-7
    var m0 = Int(scales_raw.unsafe_load[width=1](offset=4).value()) & 0x3F
    var m1 = Int(scales_raw.unsafe_load[width=1](offset=5).value()) & 0x3F
    var m2 = Int(scales_raw.unsafe_load[width=1](offset=6).value()) & 0x3F
    var m3 = Int(scales_raw.unsafe_load[width=1](offset=7).value()) & 0x3F

    # Mins 4-7: packed with upper bits
    var m4 = (Int(scales_raw.unsafe_load[width=1](offset=12).value()) & 0x0F) | \
             ((Int(scales_raw.unsafe_load[width=1](offset=4).value()) >> 6) << 4)
    var m5 = (Int(scales_raw.unsafe_load[width=1](offset=13).value()) & 0x0F) | \
             ((Int(scales_raw.unsafe_load[width=1](offset=5).value()) >> 6) << 4)
    var m6 = (Int(scales_raw.unsafe_load[width=1](offset=14).value()) & 0x0F) | \
             ((Int(scales_raw.unsafe_load[width=1](offset=6).value()) >> 6) << 4)
    var m7 = (Int(scales_raw.unsafe_load[width=1](offset=15).value()) & 0x0F) | \
             ((Int(scales_raw.unsafe_load[width=1](offset=7).value()) >> 6) << 4)

    # Pre-compute bias from Q8_K bsums in ONE operation
    var q8_bsums = q8_data.unsafe_offset(260).unsafe_bitcast[Scalar[DType.int16]]()
    var bsum0 = Int32(q8_bsums.unsafe_offset(0).unsafe_load()) + Int32(q8_bsums.unsafe_offset(1).unsafe_load())
    var bsum1 = Int32(q8_bsums.unsafe_offset(2).unsafe_load()) + Int32(q8_bsums.unsafe_offset(3).unsafe_load())
    var bsum2 = Int32(q8_bsums.unsafe_offset(4).unsafe_load()) + Int32(q8_bsums.unsafe_offset(5).unsafe_load())
    var bsum3 = Int32(q8_bsums.unsafe_offset(6).unsafe_load()) + Int32(q8_bsums.unsafe_offset(7).unsafe_load())
    var bsum4 = Int32(q8_bsums.unsafe_offset(8).unsafe_load()) + Int32(q8_bsums.unsafe_offset(9).unsafe_load())
    var bsum5 = Int32(q8_bsums.unsafe_offset(10).unsafe_load()) + Int32(q8_bsums.unsafe_offset(11).unsafe_load())
    var bsum6 = Int32(q8_bsums.unsafe_offset(12).unsafe_load()) + Int32(q8_bsums.unsafe_offset(13).unsafe_load())
    var bsum7 = Int32(q8_bsums.unsafe_offset(14).unsafe_load()) + Int32(q8_bsums.unsafe_offset(15).unsafe_load())

    var bias = dmin * q8_d * Float32(
        Int32(m0) * bsum0 + Int32(m1) * bsum1 + Int32(m2) * bsum2 + Int32(m3) * bsum3 +
        Int32(m4) * bsum4 + Int32(m5) * bsum5 + Int32(m6) * bsum6 + Int32(m7) * bsum7
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

    sumi += neon_addv(dot_0) * Int32(sc0) + neon_addv(dot_1) * Int32(sc4)

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

    sumi += neon_addv(dot_0) * Int32(sc1) + neon_addv(dot_1) * Int32(sc5)

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

    sumi += neon_addv(dot_0) * Int32(sc2) + neon_addv(dot_1) * Int32(sc6)

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

    sumi += neon_addv(dot_0) * Int32(sc3) + neon_addv(dot_1) * Int32(sc7)

    # Apply super-block scale
    return d * q8_d * Float32(sumi) - bias
