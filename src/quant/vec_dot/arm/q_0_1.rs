use std::arch::aarch64::*;

use crate::quant::types::{BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockQ8_1, QK4_0, QK8_0};

use super::fp::fp16_to_fp32;

/// Manual vdotq_s32 implementation using native SDOT instruction
/// vdotq_s32 is unstable in Rust, so we use inline assembly
/// SDOT computes: result[i] = acc[i] + sum(a[4*i..4*i+4] * b[4*i..4*i+4]) for i in 0..4
/// This is ~4x faster than the vmull+vpaddl sequence.
#[target_feature(enable = "neon,dotprod")]
unsafe fn vdotq_s32_manual(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    use std::arch::asm;

    let mut result = acc;
    asm!(
        "sdot {0}.4s, {1}.16b, {2}.16b",
        inout(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, preserves_flags)
    );
    result
}

/// Q4_0 × Q8_0 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q4_0_q8_0_neon(n: usize, x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / QK4_0;
    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);

    let m4b = vdupq_n_u8(0x0F);
    let s8b = vdupq_n_s8(0x8);

    // Process 2 blocks at a time
    let chunks = nb / 2;
    for i in 0..chunks {
        let ib = i * 2;
        let x0 = &x[ib];
        let x1 = &x[ib + 1];
        let y0 = &y[ib];
        let y1 = &y[ib + 1];

        // Load 4-bit quants
        let v0_0 = vld1q_u8(x0.qs.as_ptr());
        let v0_1 = vld1q_u8(x1.qs.as_ptr());

        // Unpack 4-bit to 8-bit
        let v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        let v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        let v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        let v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // Subtract 8 (bias)
        let v0_0ls = vsubq_s8(v0_0l, s8b);
        let v0_0hs = vsubq_s8(v0_0h, s8b);
        let v0_1ls = vsubq_s8(v0_1l, s8b);
        let v0_1hs = vsubq_s8(v0_1h, s8b);

        // Load Q8 values
        let v1_0l = vld1q_s8(y0.qs.as_ptr());
        let v1_0h = vld1q_s8(y0.qs.as_ptr().add(16));
        let v1_1l = vld1q_s8(y1.qs.as_ptr());
        let v1_1h = vld1q_s8(y1.qs.as_ptr().add(16));

        // Dot product
        let p_0 = vdotq_s32_manual(vdotq_s32_manual(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
        let p_1 = vdotq_s32_manual(vdotq_s32_manual(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

        // Multiply by scales
        let d0 = fp16_to_fp32(x0.d) * fp16_to_fp32(y0.d);
        let d1 = fp16_to_fp32(x1.d) * fp16_to_fp32(y1.d);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
    }

    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

    // Handle remainder blocks
    for ib in (chunks * 2)..nb {
        let x_b = &x[ib];
        let y_b = &y[ib];
        let d = fp16_to_fp32(x_b.d) * fp16_to_fp32(y_b.d);

        let mut sumi0 = 0i32;
        let mut sumi1 = 0i32;

        for j in 0..16 {
            let v0 = (x_b.qs[j] & 0x0F) as i32 - 8;
            let v1 = (x_b.qs[j] >> 4) as i32 - 8;

            sumi0 += v0 * y_b.qs[j] as i32;
            sumi1 += v1 * y_b.qs[j + 16] as i32;
        }

        sumf += (sumi0 + sumi1) as f32 * d;
    }

    sumf
}

/// Q8_0 × Q8_0 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q8_0_q8_0_neon(n: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / QK8_0;
    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);

    // Process 2 blocks at a time
    let chunks = nb / 2;
    for i in 0..chunks {
        let ib = i * 2;
        let x0 = &x[ib];
        let x1 = &x[ib + 1];
        let y0 = &y[ib];
        let y1 = &y[ib + 1];

        // Load Q8 values
        let x0_l = vld1q_s8(x0.qs.as_ptr());
        let x0_h = vld1q_s8(x0.qs.as_ptr().add(16));
        let x1_l = vld1q_s8(x1.qs.as_ptr());
        let x1_h = vld1q_s8(x1.qs.as_ptr().add(16));

        let y0_l = vld1q_s8(y0.qs.as_ptr());
        let y0_h = vld1q_s8(y0.qs.as_ptr().add(16));
        let y1_l = vld1q_s8(y1.qs.as_ptr());
        let y1_h = vld1q_s8(y1.qs.as_ptr().add(16));

        // Dot product
        let p_0 = vdotq_s32_manual(vdotq_s32_manual(vdupq_n_s32(0), x0_l, y0_l), x0_h, y0_h);
        let p_1 = vdotq_s32_manual(vdotq_s32_manual(vdupq_n_s32(0), x1_l, y1_l), x1_h, y1_h);

        // Multiply by scales
        let d0 = fp16_to_fp32(x0.d) * fp16_to_fp32(y0.d);
        let d1 = fp16_to_fp32(x1.d) * fp16_to_fp32(y1.d);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
    }

    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

    // Handle remainder blocks
    for ib in (chunks * 2)..nb {
        let x_b = &x[ib];
        let y_b = &y[ib];
        let d = fp16_to_fp32(x_b.d) * fp16_to_fp32(y_b.d);

        let isum: i32 = x_b.qs.iter().zip(y_b.qs.iter())
            .map(|(&a, &b)| a as i32 * b as i32)
            .sum();

        sumf += isum as f32 * d;
    }

    sumf
}

/// Q4_1 × Q8_1 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q4_1_q8_1_neon(n: usize, x: &[BlockQ4_1], y: &[BlockQ8_1]) -> f32 {
    let nb = n / QK4_0;
    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);
    let mut summs = 0.0f32;

    let m4b = vdupq_n_u8(0x0F);

    // Process 2 blocks at a time
    let chunks = nb / 2;
    for i in 0..chunks {
        let ib = i * 2;
        let x0 = &x[ib];
        let x1 = &x[ib + 1];
        let y0 = &y[ib];
        let y1 = &y[ib + 1];

        // Add min * sum term
        let m0 = fp16_to_fp32(x0.m) * fp16_to_fp32(y0.s);
        let m1 = fp16_to_fp32(x1.m) * fp16_to_fp32(y1.s);
        summs += m0 + m1;

        // Load 4-bit quants
        let v0_0 = vld1q_u8(x0.qs.as_ptr());
        let v0_1 = vld1q_u8(x1.qs.as_ptr());

        // Unpack 4-bit to 8-bit (no bias for Q4_1)
        let v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        let v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        let v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        let v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // Load Q8 values
        let v1_0l = vld1q_s8(y0.qs.as_ptr());
        let v1_0h = vld1q_s8(y0.qs.as_ptr().add(16));
        let v1_1l = vld1q_s8(y1.qs.as_ptr());
        let v1_1h = vld1q_s8(y1.qs.as_ptr().add(16));

        // Dot product
        let p_0 = vdotq_s32_manual(vdotq_s32_manual(vdupq_n_s32(0), v0_0l, v1_0l), v0_0h, v1_0h);
        let p_1 = vdotq_s32_manual(vdotq_s32_manual(vdupq_n_s32(0), v0_1l, v1_1l), v0_1h, v1_1h);

        // Multiply by scales
        let d0 = fp16_to_fp32(x0.d) * fp16_to_fp32(y0.d);
        let d1 = fp16_to_fp32(x1.d) * fp16_to_fp32(y1.d);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
    }

    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs;

    // Handle remainder blocks
    for ib in (chunks * 2)..nb {
        let x_b = &x[ib];
        let y_b = &y[ib];
        let d = fp16_to_fp32(x_b.d) * fp16_to_fp32(y_b.d);
        let m = fp16_to_fp32(x_b.m) * fp16_to_fp32(y_b.s);

        let mut isum = 0i32;
        for j in 0..16 {
            let v0 = (x_b.qs[j] & 0x0F) as i32;
            let v1 = (x_b.qs[j] >> 4) as i32;

            isum += v0 * y_b.qs[j] as i32;
            isum += v1 * y_b.qs[j + 16] as i32;
        }

        sumf += isum as f32 * d + m;
    }

    sumf
}

/// Table for bit expansion: expand 8 bits to 8 bytes
/// For each bit: if bit is 0, output byte is 0x10; if bit is 1, output byte is 0x00
#[inline(always)]
const fn build_table_b2b_1() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut i = 0;
    while i < 256 {
        let mut val: u64 = 0;
        let mut j = 0;
        while j < 8 {
            let bit = (i >> j) & 1;
            // (!bit) << 4: if bit=0, result=0x10; if bit=1, result=0x00
            let byte = if bit == 0 { 0x10u64 } else { 0x00u64 };
            val |= byte << (j * 8);
            j += 1;
        }
        table[i] = val;
        i += 1;
    }
    table
}

static TABLE_B2B_1: [u64; 256] = build_table_b2b_1();

/// Q5_0 × Q8_0 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q5_0_q8_0_neon(n: usize, x: &[BlockQ5_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / QK4_0;
    let m4b = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);

    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);

    // Process 2 blocks at a time
    let chunks = nb / 2;
    for i in 0..chunks {
        let ib = i * 2;
        let x0 = &x[ib];
        let x1 = &x[ib + 1];
        let y0 = &y[ib];
        let y1 = &y[ib + 1];

        // Load 4-bit quants
        let v0_0 = vld1q_u8(x0.qs.as_ptr());
        let v0_1 = vld1q_u8(x1.qs.as_ptr());

        // Unpack 4-bit to 8-bit
        let v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        let v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        let v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        let v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // Extract 5th bit from qh (4 bytes = 32 bits for 32 elements)
        let qh0 = u32::from_le_bytes(x0.qh);
        let qh1 = u32::from_le_bytes(x1.qh);

        // Use lookup table to expand bits (unsafe for no bounds check)
        let table_ptr = TABLE_B2B_1.as_ptr();
        let tmp0: [u64; 4] = [
            *table_ptr.add(((qh0 >> 0) & 0xFF) as usize),
            *table_ptr.add(((qh0 >> 8) & 0xFF) as usize),
            *table_ptr.add(((qh0 >> 16) & 0xFF) as usize),
            *table_ptr.add((qh0 >> 24) as usize),
        ];
        let tmp1: [u64; 4] = [
            *table_ptr.add(((qh1 >> 0) & 0xFF) as usize),
            *table_ptr.add(((qh1 >> 8) & 0xFF) as usize),
            *table_ptr.add(((qh1 >> 16) & 0xFF) as usize),
            *table_ptr.add((qh1 >> 24) as usize),
        ];

        // Load expanded bits as int8 vectors
        let qhl0 = vld1q_s8(tmp0.as_ptr() as *const i8);
        let qhh0 = vld1q_s8(tmp0.as_ptr().add(2) as *const i8);
        let qhl1 = vld1q_s8(tmp1.as_ptr() as *const i8);
        let qhh1 = vld1q_s8(tmp1.as_ptr().add(2) as *const i8);

        // Add high bit and sub 16: result = low4bit - high_bit_expanded
        let v0_0lf = vsubq_s8(v0_0l, qhl0);
        let v0_0hf = vsubq_s8(v0_0h, qhh0);
        let v0_1lf = vsubq_s8(v0_1l, qhl1);
        let v0_1hf = vsubq_s8(v0_1h, qhh1);

        // Load Q8 values
        let v1_0l = vld1q_s8(y0.qs.as_ptr());
        let v1_0h = vld1q_s8(y0.qs.as_ptr().add(16));
        let v1_1l = vld1q_s8(y1.qs.as_ptr());
        let v1_1h = vld1q_s8(y1.qs.as_ptr().add(16));

        // Dot product
        let p_0 = vaddq_s32(
            vdotq_s32_manual(vzero, v0_0lf, v1_0l),
            vdotq_s32_manual(vzero, v0_0hf, v1_0h)
        );
        let p_1 = vaddq_s32(
            vdotq_s32_manual(vzero, v0_1lf, v1_1l),
            vdotq_s32_manual(vzero, v0_1hf, v1_1h)
        );

        // Multiply by scales
        let d0 = fp16_to_fp32(x0.d) * fp16_to_fp32(y0.d);
        let d1 = fp16_to_fp32(x1.d) * fp16_to_fp32(y1.d);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
    }

    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

    // Handle remainder blocks
    for ib in (chunks * 2)..nb {
        let x_b = &x[ib];
        let y_b = &y[ib];
        let d = fp16_to_fp32(x_b.d) * fp16_to_fp32(y_b.d);

        let qh = u32::from_le_bytes(x_b.qh);
        let mut isum = 0i32;

        for j in 0..16 {
            // Bits 0-15 for low nibbles, bits 16-31 for high nibbles
            let bit_lo = ((qh >> j) & 1) as i32;
            let bit_hi = ((qh >> (j + 16)) & 1) as i32;

            let v0 = ((x_b.qs[j] & 0x0F) as i32 | (bit_lo << 4)) - 16;
            let v1 = ((x_b.qs[j] >> 4) as i32 | (bit_hi << 4)) - 16;

            isum += v0 * y_b.qs[j] as i32;
            isum += v1 * y_b.qs[j + 16] as i32;
        }

        sumf += isum as f32 * d;
    }

    sumf
}

/// Q5_1 × Q8_1 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q5_1_q8_1_neon(n: usize, x: &[BlockQ5_1], y: &[BlockQ8_1]) -> f32 {
    let nb = n / QK4_0;
    let m4b = vdupq_n_u8(0x0F);
    let vzero = vdupq_n_s32(0);

    let mut sumv0 = vdupq_n_f32(0.0);
    let mut sumv1 = vdupq_n_f32(0.0);
    let mut summs = 0.0f32;

    // Process 2 blocks at a time
    let chunks = nb / 2;
    for i in 0..chunks {
        let ib = i * 2;
        let x0 = &x[ib];
        let x1 = &x[ib + 1];
        let y0 = &y[ib];
        let y1 = &y[ib + 1];

        // Add min * sum term
        let m0 = fp16_to_fp32(x0.m) * fp16_to_fp32(y0.s);
        let m1 = fp16_to_fp32(x1.m) * fp16_to_fp32(y1.s);
        summs += m0 + m1;

        // Load 4-bit quants
        let v0_0 = vld1q_u8(x0.qs.as_ptr());
        let v0_1 = vld1q_u8(x1.qs.as_ptr());

        // Unpack 4-bit to 8-bit (no bias for Q5_1)
        let v0_0l = vreinterpretq_s8_u8(vandq_u8(v0_0, m4b));
        let v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        let v0_1l = vreinterpretq_s8_u8(vandq_u8(v0_1, m4b));
        let v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // Extract 5th bit
        let qh0 = u32::from_le_bytes(x0.qh);
        let qh1 = u32::from_le_bytes(x1.qh);

        // Use lookup table to expand bits (unsafe for no bounds check)
        let table_ptr = TABLE_B2B_1.as_ptr();
        let tmp0: [u64; 4] = [
            *table_ptr.add(((qh0 >> 0) & 0xFF) as usize),
            *table_ptr.add(((qh0 >> 8) & 0xFF) as usize),
            *table_ptr.add(((qh0 >> 16) & 0xFF) as usize),
            *table_ptr.add((qh0 >> 24) as usize),
        ];
        let tmp1: [u64; 4] = [
            *table_ptr.add(((qh1 >> 0) & 0xFF) as usize),
            *table_ptr.add(((qh1 >> 8) & 0xFF) as usize),
            *table_ptr.add(((qh1 >> 16) & 0xFF) as usize),
            *table_ptr.add((qh1 >> 24) as usize),
        ];

        let qhl0 = vld1q_s8(tmp0.as_ptr() as *const i8);
        let qhh0 = vld1q_s8(tmp0.as_ptr().add(2) as *const i8);
        let qhl1 = vld1q_s8(tmp1.as_ptr() as *const i8);
        let qhh1 = vld1q_s8(tmp1.as_ptr().add(2) as *const i8);

        // Add high bit (for Q5_1, we add the bit, not subtract)
        let v0_0lf = vaddq_s8(v0_0l, qhl0);
        let v0_0hf = vaddq_s8(v0_0h, qhh0);
        let v0_1lf = vaddq_s8(v0_1l, qhl1);
        let v0_1hf = vaddq_s8(v0_1h, qhh1);

        // Load Q8 values
        let v1_0l = vld1q_s8(y0.qs.as_ptr());
        let v1_0h = vld1q_s8(y0.qs.as_ptr().add(16));
        let v1_1l = vld1q_s8(y1.qs.as_ptr());
        let v1_1h = vld1q_s8(y1.qs.as_ptr().add(16));

        // Dot product
        let p_0 = vaddq_s32(
            vdotq_s32_manual(vzero, v0_0lf, v1_0l),
            vdotq_s32_manual(vzero, v0_0hf, v1_0h)
        );
        let p_1 = vaddq_s32(
            vdotq_s32_manual(vzero, v0_1lf, v1_1l),
            vdotq_s32_manual(vzero, v0_1hf, v1_1h)
        );

        // Multiply by scales
        let d0 = fp16_to_fp32(x0.d) * fp16_to_fp32(y0.d);
        let d1 = fp16_to_fp32(x1.d) * fp16_to_fp32(y1.d);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
    }

    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs;

    // Handle remainder blocks
    for ib in (chunks * 2)..nb {
        let x_b = &x[ib];
        let y_b = &y[ib];
        let d = fp16_to_fp32(x_b.d) * fp16_to_fp32(y_b.d);
        let m = fp16_to_fp32(x_b.m) * fp16_to_fp32(y_b.s);

        let qh = u32::from_le_bytes(x_b.qh);
        let mut isum = 0i32;

        for j in 0..16 {
            // Bits 0-15 for low nibbles, bits 16-31 for high nibbles
            let bit_lo = ((qh >> j) & 1) as i32;
            let bit_hi = ((qh >> (j + 16)) & 1) as i32;

            let v0 = (x_b.qs[j] & 0x0F) as i32 + (bit_lo << 4);
            let v1 = (x_b.qs[j] >> 4) as i32 + (bit_hi << 4);

            isum += v0 * y_b.qs[j] as i32;
            isum += v1 * y_b.qs[j + 16] as i32;
        }

        sumf += isum as f32 * d + m;
    }

    sumf
}
