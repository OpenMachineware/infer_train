# KV Cache quantization/dequantization - NEON SIMD optimized
#
# Performance target: exceed llama.cpp scalar implementation
#
# Block sizes:
#   Q4_0: 18 bytes (d:2 + qs:16)
#   Q4_1: 20 bytes (d:2 + m:2 + qs:16)
#   Q5_0: 22 bytes (d:2 + qh:4 + qs:16)
#   Q5_1: 24 bytes (d:2 + m:2 + qh:4 + qs:16)
#   Q8_0: 34 bytes (d:2 + qs:32)

from src.core.ops.cpu.simd.simd_neon import (
    neon_ld1_u8_x2, neon_ld1_s8_x2,
    neon_vmovl_u8, neon_vmovl_s8, neon_vmovl_high_s8,
    neon_vmovl_s16, neon_vmovl_high_s16,
    neon_vget_low_u8, neon_vget_high_u8,
    neon_vget_low_s8, neon_vget_high_s8,
    neon_vget_low_s16, neon_vget_high_s16,
    neon_vcvtq_f32_s32,
    neon_tbl1,
    neon_vreinterpretq_s16_u16,
    neon_vmaxq_f32,
    neon_vmaxvq_f32,
    neon_vabsq_f32,
    neon_vcvtnq_s32_f32,
    neon_vqmovn_s32,
)
from std.memory import Pointer
from std.origin import MutUntrackedOrigin
from std.math import round

# Block constants
comptime KV_QK = 32


# ============================================================================
# Helper functions
# ============================================================================

@always_inline
def load_u8x16(ptr: Pointer[UInt8, MutUntrackedOrigin]) -> SIMD[DType.uint8, 16]:
    """Load 16 bytes into a SIMD vector.

    Uses neon_ld1_u8_x2 which loads 32 bytes and returns .lo (first 16).
    This is more efficient than 16 scalar loads.
    """
    var vec_pair = neon_ld1_u8_x2(ptr)
    return vec_pair.lo


# ============================================================================
# Q4_0 SIMD (18 bytes: d:2 + qs:16) - Vectorized store
# ============================================================================

def dequantize_row_q4_0_neon(
    src: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Scalar[DType.float16], MutUntrackedOrigin],
    n: Int,
):
    """Dequantize Q4_0 blocks into fp16 using NEON SIMD with vectorized store.

    Formula: value = d * (q - 8)
    """
    var nb = n // KV_QK
    var src_off = 0
    var dst_off = 0

    for _ in range(nb):
        # Load scale (fp16)
        var d_f16 = src.unsafe_offset(src_off).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0)

        # Load qs (16 bytes = 32 packed 4-bit values)
        var qs = load_u8x16(src.unsafe_offset(src_off + 2))

        # Unpack 4-bit values using SIMD
        var mask4 = SIMD[DType.uint8, 16](15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15)
        var qs_lo = qs & mask4
        var qs_hi = (qs >> 4) & mask4

        # Widen uint8 -> uint16
        var lo_low = neon_vget_low_u8(qs_lo)
        var lo_high = neon_vget_high_u8(qs_lo)
        var hi_low = neon_vget_low_u8(qs_hi)
        var hi_high = neon_vget_high_u8(qs_hi)

        var lo_low_16 = neon_vmovl_u8(lo_low)
        var lo_high_16 = neon_vmovl_u8(lo_high)
        var hi_low_16 = neon_vmovl_u8(hi_low)
        var hi_high_16 = neon_vmovl_u8(hi_high)

        # Vectorized: bias -8, scale, convert to float16
        var bias = SIMD[DType.int16, 8](-8, -8, -8, -8, -8, -8, -8, -8)
        var d_vec = SIMD[DType.float32, 8](Float32(d_f16))

        # int16 values with bias
        var q0 = neon_vreinterpretq_s16_u16(lo_low_16) + bias
        var q1 = neon_vreinterpretq_s16_u16(hi_low_16) + bias
        var q2 = neon_vreinterpretq_s16_u16(lo_high_16) + bias
        var q3 = neon_vreinterpretq_s16_u16(hi_high_16) + bias

        # Cast to float32, multiply, cast to float16
        var f0 = q0.cast[DType.float32]() * d_vec
        var f1 = q1.cast[DType.float32]() * d_vec
        var f2 = q2.cast[DType.float32]() * d_vec
        var f3 = q3.cast[DType.float32]() * d_vec

        var h0 = f0.cast[DType.float16]()
        var h1 = f1.cast[DType.float16]()
        var h2 = f2.cast[DType.float16]()
        var h3 = f3.cast[DType.float16]()

        # Interleave: low nibble at even, high at odd positions
        var even0 = SIMD[DType.float16, 8](h0[0], h1[0], h0[1], h1[1], h0[2], h1[2], h0[3], h1[3])
        var odd0 = SIMD[DType.float16, 8](h0[4], h1[4], h0[5], h1[5], h0[6], h1[6], h0[7], h1[7])
        var even1 = SIMD[DType.float16, 8](h2[0], h3[0], h2[1], h3[1], h2[2], h3[2], h2[3], h3[3])
        var odd1 = SIMD[DType.float16, 8](h2[4], h3[4], h2[5], h3[5], h2[6], h3[6], h2[7], h3[7])

        dst.unsafe_store[width=8](offset=dst_off, val=even0)
        dst.unsafe_store[width=8](offset=dst_off + 8, val=odd0)
        dst.unsafe_store[width=8](offset=dst_off + 16, val=even1)
        dst.unsafe_store[width=8](offset=dst_off + 24, val=odd1)

        src_off += 18
        dst_off += KV_QK


# ============================================================================
# Q8_0 SIMD (34 bytes: d:2 + qs:32) - Float32 output for speed
# ============================================================================

def dequantize_row_q8_0_f32_neon(
    src: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Scalar[DType.float32], MutUntrackedOrigin],
    n: Int,
):
    """Dequantize Q8_0 blocks into fp32 using NEON SIMD with two-stage conversion.

    Formula: value = d * q

    Matches llama.cpp NEON implementation:
    1. int8 -> int16 (vmovl_s8 / sshll)
    2. int16 -> int32 (vmovl_s16 / sshll)
    3. int32 -> float32 (vcvtq_f32_s32 / scvtf)
    """
    var nb = n // KV_QK
    var src_off = 0
    var dst_off = 0

    for _ in range(nb):
        var d_f16 = src.unsafe_offset(src_off).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0)
        var d = Float32(d_f16)

        # Load 32 signed int8 values
        var qs_pair = neon_ld1_s8_x2(src.unsafe_offset(src_off + 2))
        var qs0 = qs_pair.lo  # int8x16
        var qs1 = qs_pair.hi

        # Stage 1: int8 -> int16
        var w0 = neon_vmovl_s8(neon_vget_low_s8(qs0))   # int16x8
        var w1 = neon_vmovl_high_s8(qs0)

        # Stage 2: int16 -> int32
        var i0_low = neon_vmovl_s16(neon_vget_low_s16(w0))    # int32x4
        var i0_high = neon_vmovl_high_s16(w0)
        var i1_low = neon_vmovl_s16(neon_vget_low_s16(w1))
        var i1_high = neon_vmovl_high_s16(w1)

        # Stage 3: int32 -> float32 and multiply by scale
        var f0 = neon_vcvtq_f32_s32(i0_low) * SIMD[DType.float32, 4](d)
        var f1 = neon_vcvtq_f32_s32(i0_high) * SIMD[DType.float32, 4](d)
        var f2 = neon_vcvtq_f32_s32(i1_low) * SIMD[DType.float32, 4](d)
        var f3 = neon_vcvtq_f32_s32(i1_high) * SIMD[DType.float32, 4](d)

        dst.unsafe_store[width=4](offset=dst_off, val=f0)
        dst.unsafe_store[width=4](offset=dst_off + 4, val=f1)
        dst.unsafe_store[width=4](offset=dst_off + 8, val=f2)
        dst.unsafe_store[width=4](offset=dst_off + 12, val=f3)

        # Process second 16 bytes
        var w2 = neon_vmovl_s8(neon_vget_low_s8(qs1))
        var w3 = neon_vmovl_high_s8(qs1)

        var i2_low = neon_vmovl_s16(neon_vget_low_s16(w2))
        var i2_high = neon_vmovl_high_s16(w2)
        var i3_low = neon_vmovl_s16(neon_vget_low_s16(w3))
        var i3_high = neon_vmovl_high_s16(w3)

        var f4 = neon_vcvtq_f32_s32(i2_low) * SIMD[DType.float32, 4](d)
        var f5 = neon_vcvtq_f32_s32(i2_high) * SIMD[DType.float32, 4](d)
        var f6 = neon_vcvtq_f32_s32(i3_low) * SIMD[DType.float32, 4](d)
        var f7 = neon_vcvtq_f32_s32(i3_high) * SIMD[DType.float32, 4](d)

        dst.unsafe_store[width=4](offset=dst_off + 16, val=f4)
        dst.unsafe_store[width=4](offset=dst_off + 20, val=f5)
        dst.unsafe_store[width=4](offset=dst_off + 24, val=f6)
        dst.unsafe_store[width=4](offset=dst_off + 28, val=f7)

        src_off += 34
        dst_off += KV_QK


def dequantize_row_q8_0_neon(
    src: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Scalar[DType.float16], MutUntrackedOrigin],
    n: Int,
):
    """Dequantize Q8_0 blocks into fp16 using NEON SIMD with vectorized store.

    Formula: value = d * q
    """
    var nb = n // KV_QK
    var src_off = 0
    var dst_off = 0

    for _ in range(nb):
        # Load scale (fp16)
        var d_f16 = src.unsafe_offset(src_off).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0)

        # Load 32 int8 values using ld1.16b
        var qs_pair = neon_ld1_u8_x2(src.unsafe_offset(src_off + 2))
        var qs0 = qs_pair.lo
        var qs1 = qs_pair.hi

        # Widen to int16 (signed)
        var qs0_low = neon_vget_low_u8(qs0)
        var qs0_high = neon_vget_high_u8(qs0)
        var qs1_low = neon_vget_low_u8(qs1)
        var qs1_high = neon_vget_high_u8(qs1)

        var q0_low_16 = neon_vreinterpretq_s16_u16(neon_vmovl_u8(qs0_low))
        var q0_high_16 = neon_vreinterpretq_s16_u16(neon_vmovl_u8(qs0_high))
        var q1_low_16 = neon_vreinterpretq_s16_u16(neon_vmovl_u8(qs1_low))
        var q1_high_16 = neon_vreinterpretq_s16_u16(neon_vmovl_u8(qs1_high))

        # Convert to float32, scale, then to float16
        var d_vec = SIMD[DType.float32, 8](Float32(d_f16))

        var f0_low = q0_low_16.cast[DType.float32]() * d_vec
        var f0_high = q0_high_16.cast[DType.float32]() * d_vec
        var f1_low = q1_low_16.cast[DType.float32]() * d_vec
        var f1_high = q1_high_16.cast[DType.float32]() * d_vec

        var h0 = f0_low.cast[DType.float16]()
        var h1 = f0_high.cast[DType.float16]()
        var h2 = f1_low.cast[DType.float16]()
        var h3 = f1_high.cast[DType.float16]()

        dst.unsafe_store[width=8](offset=dst_off, val=h0)
        dst.unsafe_store[width=8](offset=dst_off + 8, val=h1)
        dst.unsafe_store[width=8](offset=dst_off + 16, val=h2)
        dst.unsafe_store[width=8](offset=dst_off + 24, val=h3)

        src_off += 34
        dst_off += KV_QK


# ============================================================================
# Q5_0 SIMD (22 bytes: d:2 + qh:4 + qs:16) - Vectorized
# ============================================================================

def dequantize_row_q5_0_neon(
    src: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Scalar[DType.float16], MutUntrackedOrigin],
    n: Int,
):
    """Dequantize Q5_0 blocks into fp16 using NEON SIMD with vectorized store.

    Q5_0 layout: d(fp16) + qh(4 bytes) + qs(16 bytes)
    Formula: value = d * (q - 16)

    Strategy: Process 8 values at a time using SIMD vectors.
    """
    var nb = n // KV_QK
    var src_off = 0
    var dst_off = 0

    for _ in range(nb):
        var d_f16 = src.unsafe_offset(src_off).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0)
        var d = Float32(d_f16)

        # Load qh (4 bytes = 32 bits)
        var qh_bytes = src.unsafe_offset(src_off + 2)

        # Load qs and unpack 4-bit values
        var qs = load_u8x16(src.unsafe_offset(src_off + 6))
        var mask4 = SIMD[DType.uint8, 16](15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15)
        var qs_lo = qs & mask4  # Low nibbles
        var qs_hi = (qs >> 4) & mask4  # High nibbles

        # Process 4 groups of 8 values each
        # Group 0: positions 0-7, uses qs_lo[0-7], qh_byte[0] bits 0-7
        # Group 1: positions 8-15, uses qs_lo[8-15], qh_byte[1] bits 0-7
        # Group 2: positions 16-23, uses qs_hi[0-7], qh_byte[2] bits 0-7
        # Group 3: positions 24-31, uses qs_hi[8-15], qh_byte[3] bits 0-7

        var d_vec = SIMD[DType.float32, 8](d)
        var bias = SIMD[DType.int16, 8](16, 16, 16, 16, 16, 16, 16, 16)

        for g in range(4):
            var qh_byte = Int(qh_bytes.unsafe_load[width=1](offset=g))

            # Extract 8 qh bits and place into int16 vector
            var qh_int = Int(qh_byte)
            var qh_bits = SIMD[DType.int16, 8](Int16((qh_int >> 0) & 1))
            qh_bits[1] = Int16((qh_int >> 1) & 1)
            qh_bits[2] = Int16((qh_int >> 2) & 1)
            qh_bits[3] = Int16((qh_int >> 3) & 1)
            qh_bits[4] = Int16((qh_int >> 4) & 1)
            qh_bits[5] = Int16((qh_int >> 5) & 1)
            qh_bits[6] = Int16((qh_int >> 6) & 1)
            qh_bits[7] = Int16((qh_int >> 7) & 1)

            # Get 4-bit values
            var q4_u8: SIMD[DType.uint8, 8]
            if g == 0:
                q4_u8 = neon_vget_low_u8(qs_lo)
            elif g == 1:
                q4_u8 = neon_vget_high_u8(qs_lo)
            elif g == 2:
                q4_u8 = neon_vget_low_u8(qs_hi)
            else:
                q4_u8 = neon_vget_high_u8(qs_hi)

            # Widen to int16
            var q4_16 = neon_vreinterpretq_s16_u16(neon_vmovl_u8(q4_u8))

            # Combine: q = q4 | (qh_bit << 4)
            var q5 = q4_16 + (qh_bits << 4) - bias

            # Convert to float32, scale, convert to float16
            var f = q5.cast[DType.float32]() * d_vec
            var h = f.cast[DType.float16]()

            dst.unsafe_store[width=8](offset=dst_off + g * 8, val=h)

        src_off += 22
        dst_off += KV_QK


# ============================================================================
# Q4_1 SIMD (20 bytes: d:2 + m:2 + qs:16) - Vectorized store
# ============================================================================

def dequantize_row_q4_1_neon(
    src: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Scalar[DType.float16], MutUntrackedOrigin],
    n: Int,
):
    """Dequantize Q4_1 blocks into fp16 using NEON SIMD with vectorized store.

    Q4_1 layout: d(fp16) + m(fp16) + qs(16 bytes)
    Formula: value = d * q + m
    """
    var nb = n // KV_QK
    var src_off = 0
    var dst_off = 0

    for _ in range(nb):
        var d_f16 = src.unsafe_offset(src_off).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0)
        var m_f16 = src.unsafe_offset(src_off + 2).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0)

        var qs = load_u8x16(src.unsafe_offset(src_off + 4))

        # Unpack 4-bit values
        var mask4 = SIMD[DType.uint8, 16](15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15)
        var qs_lo = qs & mask4
        var qs_hi = (qs >> 4) & mask4

        # Widen to uint16
        var lo_low = neon_vget_low_u8(qs_lo)
        var lo_high = neon_vget_high_u8(qs_lo)
        var hi_low = neon_vget_low_u8(qs_hi)
        var hi_high = neon_vget_high_u8(qs_hi)

        var lo_low_16 = neon_vmovl_u8(lo_low)
        var lo_high_16 = neon_vmovl_u8(lo_high)
        var hi_low_16 = neon_vmovl_u8(hi_low)
        var hi_high_16 = neon_vmovl_u8(hi_high)

        # Vectorized: d * q + m
        var d_vec = SIMD[DType.float32, 8](Float32(d_f16))
        var m_vec = SIMD[DType.float32, 8](Float32(m_f16))

        # int16 -> float32, compute, -> float16
        var q0 = neon_vreinterpretq_s16_u16(lo_low_16)
        var q1 = neon_vreinterpretq_s16_u16(hi_low_16)
        var q2 = neon_vreinterpretq_s16_u16(lo_high_16)
        var q3 = neon_vreinterpretq_s16_u16(hi_high_16)

        var f0 = q0.cast[DType.float32]() * d_vec + m_vec
        var f1 = q1.cast[DType.float32]() * d_vec + m_vec
        var f2 = q2.cast[DType.float32]() * d_vec + m_vec
        var f3 = q3.cast[DType.float32]() * d_vec + m_vec

        var h0 = f0.cast[DType.float16]()
        var h1 = f1.cast[DType.float16]()
        var h2 = f2.cast[DType.float16]()
        var h3 = f3.cast[DType.float16]()

        # Interleave: low nibble at even, high at odd
        var even0 = SIMD[DType.float16, 8](h0[0], h1[0], h0[1], h1[1], h0[2], h1[2], h0[3], h1[3])
        var odd0 = SIMD[DType.float16, 8](h0[4], h1[4], h0[5], h1[5], h0[6], h1[6], h0[7], h1[7])
        var even1 = SIMD[DType.float16, 8](h2[0], h3[0], h2[1], h3[1], h2[2], h3[2], h2[3], h3[3])
        var odd1 = SIMD[DType.float16, 8](h2[4], h3[4], h2[5], h3[5], h2[6], h3[6], h2[7], h3[7])

        dst.unsafe_store[width=8](offset=dst_off, val=even0)
        dst.unsafe_store[width=8](offset=dst_off + 8, val=odd0)
        dst.unsafe_store[width=8](offset=dst_off + 16, val=even1)
        dst.unsafe_store[width=8](offset=dst_off + 24, val=odd1)

        src_off += 20
        dst_off += KV_QK


# ============================================================================
# Q5_1 SIMD (24 bytes: d:2 + m:2 + qh:4 + qs:16)
# ============================================================================

def dequantize_row_q5_1_neon(
    src: Pointer[UInt8, MutUntrackedOrigin],
    dst: Pointer[Scalar[DType.float16], MutUntrackedOrigin],
    n: Int,
):
    """Dequantize Q5_1 blocks into fp16 using NEON SIMD with vectorized store.

    Q5_1 layout: d(fp16) + m(fp16) + qh(4 bytes) + qs(16 bytes)
    Formula: value = d * q + m
    """
    var nb = n // KV_QK
    var src_off = 0
    var dst_off = 0

    for _ in range(nb):
        var d_f16 = src.unsafe_offset(src_off).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0)
        var m_f16 = src.unsafe_offset(src_off + 2).unsafe_bitcast[Scalar[DType.float16]]().unsafe_load[width=1](offset=0)

        # Load qh (4 bytes = 32 bits)
        var qh_bytes = src.unsafe_offset(src_off + 4)

        # Load qs and unpack 4-bit values
        var qs = load_u8x16(src.unsafe_offset(src_off + 8))
        var mask4 = SIMD[DType.uint8, 16](15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15)
        var qs_lo = qs & mask4
        var qs_hi = (qs >> 4) & mask4

        var d_vec = SIMD[DType.float32, 8](Float32(d_f16))
        var m_vec = SIMD[DType.float32, 8](Float32(m_f16))

        for g in range(4):
            var qh_byte = Int(qh_bytes.unsafe_load[width=1](offset=g))

            # Extract 8 qh bits
            var qh_int = Int(qh_byte)
            var qh_bits = SIMD[DType.int16, 8](Int16((qh_int >> 0) & 1))
            qh_bits[1] = Int16((qh_int >> 1) & 1)
            qh_bits[2] = Int16((qh_int >> 2) & 1)
            qh_bits[3] = Int16((qh_int >> 3) & 1)
            qh_bits[4] = Int16((qh_int >> 4) & 1)
            qh_bits[5] = Int16((qh_int >> 5) & 1)
            qh_bits[6] = Int16((qh_int >> 6) & 1)
            qh_bits[7] = Int16((qh_int >> 7) & 1)

            # Get 4-bit values
            var q4_u8: SIMD[DType.uint8, 8]
            if g == 0:
                q4_u8 = neon_vget_low_u8(qs_lo)
            elif g == 1:
                q4_u8 = neon_vget_high_u8(qs_lo)
            elif g == 2:
                q4_u8 = neon_vget_low_u8(qs_hi)
            else:
                q4_u8 = neon_vget_high_u8(qs_hi)

            # Widen to int16
            var q4_16 = neon_vreinterpretq_s16_u16(neon_vmovl_u8(q4_u8))

            # Combine: q = q4 | (qh_bit << 4)
            var q5 = q4_16 + (qh_bits << 4)

            # Convert to float32, compute, convert to float16
            var f = q5.cast[DType.float32]() * d_vec + m_vec
            var h = f.cast[DType.float16]()

            dst.unsafe_store[width=8](offset=dst_off + g * 8, val=h)

        src_off += 24
        dst_off += KV_QK


# ============================================================================
# Q8_0 Quantization SIMD (32 fp16 -> 34 bytes)
# ============================================================================

def quantize_row_q8_0_neon(
    src: Pointer[Scalar[DType.float16], MutUntrackedOrigin],
    dst: Pointer[UInt8, MutUntrackedOrigin],
    n: Int,
):
    """Quantize fp16 values into Q8_0 blocks using NEON SIMD.

    Matches llama.cpp NEON implementation (quants.c:41-83):
    1. Load 32 fp16 -> 8x float32x4
    2. Tree-reduce amax using vmaxq_f32 + vmaxvq_f32
    3. Compute scale d = amax / 127
    4. Quantize: round -> clamp -> int8
    5. Store results with scale
    """
    var nb = n // KV_QK
    var src_off = 0
    var dst_off = 0

    for _ in range(nb):
        # Load 32 fp16 values, convert to fp32, split into 8 vectors of 4
        # (matching llama.cpp's 8x float32x4_t)
        var srcv0 = src.unsafe_load[width=4](offset=src_off).cast[DType.float32]()
        var srcv1 = src.unsafe_load[width=4](offset=src_off + 4).cast[DType.float32]()
        var srcv2 = src.unsafe_load[width=4](offset=src_off + 8).cast[DType.float32]()
        var srcv3 = src.unsafe_load[width=4](offset=src_off + 12).cast[DType.float32]()
        var srcv4 = src.unsafe_load[width=4](offset=src_off + 16).cast[DType.float32]()
        var srcv5 = src.unsafe_load[width=4](offset=src_off + 20).cast[DType.float32]()
        var srcv6 = src.unsafe_load[width=4](offset=src_off + 24).cast[DType.float32]()
        var srcv7 = src.unsafe_load[width=4](offset=src_off + 28).cast[DType.float32]()

        # Compute abs values using vabsq_f32
        var asrcv0 = neon_vabsq_f32(srcv0)
        var asrcv1 = neon_vabsq_f32(srcv1)
        var asrcv2 = neon_vabsq_f32(srcv2)
        var asrcv3 = neon_vabsq_f32(srcv3)
        var asrcv4 = neon_vabsq_f32(srcv4)
        var asrcv5 = neon_vabsq_f32(srcv5)
        var asrcv6 = neon_vabsq_f32(srcv6)
        var asrcv7 = neon_vabsq_f32(srcv7)

        # Tree reduction for amax using vmaxq_f32 (matching llama.cpp)
        # Level 1: 8 -> 4
        var amaxv0 = neon_vmaxq_f32(asrcv0, asrcv1)
        var amaxv1 = neon_vmaxq_f32(asrcv2, asrcv3)
        var amaxv2 = neon_vmaxq_f32(asrcv4, asrcv5)
        var amaxv3 = neon_vmaxq_f32(asrcv6, asrcv7)

        # Level 2: 4 -> 2
        var amaxw0 = neon_vmaxq_f32(amaxv0, amaxv1)
        var amaxw1 = neon_vmaxq_f32(amaxv2, amaxv3)

        # Level 3: 2 -> 1
        var amaxx = neon_vmaxq_f32(amaxw0, amaxw1)

        # Final horizontal max using vmaxvq_f32
        var amax = neon_vmaxvq_f32(amaxx)

        # Compute scale
        var d = amax / Float32(127.0)
        var id = Float32(0.0)
        if d != Float32(0.0):
            id = Float32(1.0) / d

        # Store scale as fp16 (bitcast pointer)
        var d_ptr = dst.unsafe_offset(dst_off).unsafe_bitcast[Scalar[DType.float16]]()
        d_ptr.unsafe_store(0, Scalar[DType.float16](d))

        # Quantize each vector: float32 -> int8 using vcvtnq_s32_f32
        var qs_ptr = dst.unsafe_offset(dst_off + 2)
        var id_vec = SIMD[DType.float32, 4](id)

        # Process 8 vectors (matching llama.cpp)
        # Use vcvtnq_s32_f32 for hardware-accelerated rounding
        # No clamp needed: scale = amax/127 ensures values fit in [-127, 127]
        var qsv0 = neon_vcvtnq_s32_f32(srcv0 * id_vec)
        var qsv1 = neon_vcvtnq_s32_f32(srcv1 * id_vec)
        var qsv2 = neon_vcvtnq_s32_f32(srcv2 * id_vec)
        var qsv3 = neon_vcvtnq_s32_f32(srcv3 * id_vec)
        var qsv4 = neon_vcvtnq_s32_f32(srcv4 * id_vec)
        var qsv5 = neon_vcvtnq_s32_f32(srcv5 * id_vec)
        var qsv6 = neon_vcvtnq_s32_f32(srcv6 * id_vec)
        var qsv7 = neon_vcvtnq_s32_f32(srcv7 * id_vec)

        # Direct store as int8 (matching llama.cpp's vgetq_lane pattern)
        # Each value is guaranteed in [-127, 127], so lower 8 bits are correct
        qs_ptr.unsafe_store(0, UInt8(qsv0[0] & 0xFF))
        qs_ptr.unsafe_store(1, UInt8(qsv0[1] & 0xFF))
        qs_ptr.unsafe_store(2, UInt8(qsv0[2] & 0xFF))
        qs_ptr.unsafe_store(3, UInt8(qsv0[3] & 0xFF))
        qs_ptr.unsafe_store(4, UInt8(qsv1[0] & 0xFF))
        qs_ptr.unsafe_store(5, UInt8(qsv1[1] & 0xFF))
        qs_ptr.unsafe_store(6, UInt8(qsv1[2] & 0xFF))
        qs_ptr.unsafe_store(7, UInt8(qsv1[3] & 0xFF))
        qs_ptr.unsafe_store(8, UInt8(qsv2[0] & 0xFF))
        qs_ptr.unsafe_store(9, UInt8(qsv2[1] & 0xFF))
        qs_ptr.unsafe_store(10, UInt8(qsv2[2] & 0xFF))
        qs_ptr.unsafe_store(11, UInt8(qsv2[3] & 0xFF))
        qs_ptr.unsafe_store(12, UInt8(qsv3[0] & 0xFF))
        qs_ptr.unsafe_store(13, UInt8(qsv3[1] & 0xFF))
        qs_ptr.unsafe_store(14, UInt8(qsv3[2] & 0xFF))
        qs_ptr.unsafe_store(15, UInt8(qsv3[3] & 0xFF))
        qs_ptr.unsafe_store(16, UInt8(qsv4[0] & 0xFF))
        qs_ptr.unsafe_store(17, UInt8(qsv4[1] & 0xFF))
        qs_ptr.unsafe_store(18, UInt8(qsv4[2] & 0xFF))
        qs_ptr.unsafe_store(19, UInt8(qsv4[3] & 0xFF))
        qs_ptr.unsafe_store(20, UInt8(qsv5[0] & 0xFF))
        qs_ptr.unsafe_store(21, UInt8(qsv5[1] & 0xFF))
        qs_ptr.unsafe_store(22, UInt8(qsv5[2] & 0xFF))
        qs_ptr.unsafe_store(23, UInt8(qsv5[3] & 0xFF))
        qs_ptr.unsafe_store(24, UInt8(qsv6[0] & 0xFF))
        qs_ptr.unsafe_store(25, UInt8(qsv6[1] & 0xFF))
        qs_ptr.unsafe_store(26, UInt8(qsv6[2] & 0xFF))
        qs_ptr.unsafe_store(27, UInt8(qsv6[3] & 0xFF))
        qs_ptr.unsafe_store(28, UInt8(qsv7[0] & 0xFF))
        qs_ptr.unsafe_store(29, UInt8(qsv7[1] & 0xFF))
        qs_ptr.unsafe_store(30, UInt8(qsv7[2] & 0xFF))
        qs_ptr.unsafe_store(31, UInt8(qsv7[3] & 0xFF))

        src_off += KV_QK
        dst_off += 34


@always_inline
def _simd_max_pair(
    a0: Float32, a1: Float32, a2: Float32, a3: Float32,
    a4: Float32, a5: Float32, a6: Float32, a7: Float32,
) -> Float32:
    """Compute max of 8 values using pairwise reduction."""
    var m01 = max(a0, a1)
    var m23 = max(a2, a3)
    var m45 = max(a4, a5)
    var m67 = max(a6, a7)

    var m0123 = max(m01, m23)
    var m4567 = max(m45, m67)

    return max(m0123, m4567)


@always_inline
def _quantize_and_store_8(
    v: SIMD[DType.float32, 8],
    id: Float32,
    ptr: Pointer[UInt8, MutUntrackedOrigin],
    offset: Int,
):
    """Quantize 8 float32 values to int8 and store."""
    var id_vec = SIMD[DType.float32, 8](id)
    var scaled = v * id_vec

    # Round and convert to int8
    for i in range(8):
        var q = Int(round(scaled[i]))
        if q > 127:
            q = 127
        if q < -127:
            q = -127
        ptr.unsafe_store(offset + i, UInt8(q & 0xFF))


# ============================================================================
# Q4_0 Quantization SIMD (32 fp16 -> 18 bytes) - Placeholder for now
# ============================================================================

# Note: Q4_0/Q4_1/Q5_0/Q5_1 quantization SIMD implementations are complex
# due to 4-bit packing and offset handling. For now, we keep the scalar
# implementations in kv_cache.mojo which already work correctly.
#
# These can be optimized later using similar NEON intrinsics approach as Q8_0.
