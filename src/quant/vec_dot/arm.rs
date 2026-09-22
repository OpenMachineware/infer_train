use std::arch::aarch64::*;

/// FP32 vector dot product (NEON implementation)
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_fp32_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);

    // Process 4 floats at a time
    let chunks = n / 4;
    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        sum = vmlaq_f32(sum, va, vb);
    }

    // Horizontal sum
    let result = vaddvq_f32(sum);

    // Handle remainder
    let mut remainder = 0.0f32;
    for i in (chunks * 4)..n {
        remainder += a[i] * b[i];
    }

    result + remainder
}

/// FP16 vector dot product (NEON implementation)
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_fp16_neon(a: &[u16], b: &[u16]) -> f32 {
    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);

    // Process 8 FP16 at a time
    let chunks = n / 8;
    for i in 0..chunks {
        let va = vld1q_u16(a.as_ptr().add(i * 8));
        let vb = vld1q_u16(b.as_ptr().add(i * 8));

        // Convert FP16 to FP32 (lower and upper halves)
        let va_f32_lo = vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(va)));
        let va_f32_hi = vcvt_f32_f16(vget_high_f16(vreinterpretq_f16_u16(va)));
        let vb_f32_lo = vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(vb)));
        let vb_f32_hi = vcvt_f32_f16(vget_high_f16(vreinterpretq_f16_u16(vb)));

        // Multiply and accumulate
        sum = vmlaq_f32(sum, va_f32_lo, vb_f32_lo);
        sum = vmlaq_f32(sum, va_f32_hi, vb_f32_hi);
    }

    let result = vaddvq_f32(sum);

    // Handle remainder
    let mut remainder = 0.0f32;
    for i in (chunks * 8)..n {
        let a_f32 = half::f16::from_bits(a[i]).to_f32();
        let b_f32 = half::f16::from_bits(b[i]).to_f32();
        remainder += a_f32 * b_f32;
    }

    result + remainder
}

/// BF16 vector dot product (NEON implementation)
/// Note: Requires ARMv8.2-A with BF16 extension
#[cfg(target_feature = "bf16")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_bf16_neon(a: &[u16], b: &[u16]) -> f32 {
    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);

    // Process 8 BF16 at a time
    let chunks = n / 8;
    for i in 0..chunks {
        let va = vld1q_u16(a.as_ptr().add(i * 8));
        let vb = vld1q_u16(b.as_ptr().add(i * 8));

        // Convert BF16 to FP32 and multiply
        let va_f32_lo = vcvt_f32_bf16(vget_low_bf16(vreinterpretq_bf16_u16(va)));
        let va_f32_hi = vcvt_f32_bf16(vget_high_bf16(vreinterpretq_bf16_u16(va)));
        let vb_f32_lo = vcvt_f32_bf16(vget_low_bf16(vreinterpretq_bf16_u16(vb)));
        let vb_f32_hi = vcvt_f32_bf16(vget_high_bf16(vreinterpretq_bf16_u16(vb)));

        sum = vmlaq_f32(sum, va_f32_lo, vb_f32_lo);
        sum = vmlaq_f32(sum, va_f32_hi, vb_f32_hi);
    }

    let result = vaddvq_f32(sum);

    // Handle remainder
    let mut remainder = 0.0f32;
    for i in (chunks * 8)..n {
        let a_f32 = half::bf16::from_bits(a[i]).to_f32();
        let b_f32 = half::bf16::from_bits(b[i]).to_f32();
        remainder += a_f32 * b_f32;
    }

    result + remainder
}

/// BF16 vector dot product (NEON fallback without BF16 extension)
/// Uses software conversion from BF16 to FP32
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_bf16_neon_fallback(a: &[u16], b: &[u16]) -> f32 {
    let n = a.len();
    let mut sum = vdupq_n_f32(0.0);

    // Process 4 BF16 at a time using scalar conversion
    // BF16 is just the upper 16 bits of FP32, so we can use shifts
    let chunks = n / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let mut a_f32 = [0.0f32; 4];
        let mut b_f32 = [0.0f32; 4];

        for j in 0..4 {
            // BF16 to FP32: shift left 16 bits
            a_f32[j] = f32::from_bits((a[idx + j] as u32) << 16);
            b_f32[j] = f32::from_bits((b[idx + j] as u32) << 16);
        }

        let va = vld1q_f32(a_f32.as_ptr());
        let vb = vld1q_f32(b_f32.as_ptr());
        sum = vmlaq_f32(sum, va, vb);
    }

    let result = vaddvq_f32(sum);

    // Handle remainder
    let mut remainder = 0.0f32;
    for i in (chunks * 4)..n {
        let a_f32 = f32::from_bits((a[i] as u32) << 16);
        let b_f32 = f32::from_bits((b[i] as u32) << 16);
        remainder += a_f32 * b_f32;
    }

    result + remainder
}

// ===== Quantized vec_dot NEON implementations =====

use crate::quant::types::{BlockQ4_0, BlockQ4_1, BlockQ8_0, BlockQ8_1, QK4_0, QK8_0};

/// Manual vdotq_s32 implementation
/// vdotq_s32 is unstable in Rust, so we implement it manually
/// Each vmull_s8 produces int16x8_t (8 products of 16-bit)
/// We need to pairwise add them into int32x4_t
#[inline(always)]
unsafe fn vdotq_s32_manual(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    // Split into two halves
    let a_lo = vget_low_s8(a);
    let a_hi = vget_high_s8(a);
    let b_lo = vget_low_s8(b);
    let b_hi = vget_high_s8(b);

    // Widen to 16-bit and multiply: int8x8_t * int8x8_t -> int16x8_t
    let prod_lo = vmull_s8(a_lo, b_lo);
    let prod_hi = vmull_s8(a_hi, b_hi);

    // Pairwise add int16x8_t -> int32x4_t
    let sum_lo = vpaddlq_s16(prod_lo);
    let sum_hi = vpaddlq_s16(prod_hi);

    // Add to accumulator
    vaddq_s32(vaddq_s32(acc, sum_lo), sum_hi)
}

/// Q4_0 × Q8_0 vector dot product (NEON implementation)
#[target_feature(enable = "neon")]
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
        let d0 = half::f16::from_bits(x0.d).to_f32() * half::f16::from_bits(y0.d).to_f32();
        let d1 = half::f16::from_bits(x1.d).to_f32() * half::f16::from_bits(y1.d).to_f32();

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
    }

    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

    // Handle remainder blocks
    for ib in (chunks * 2)..nb {
        let x_b = &x[ib];
        let y_b = &y[ib];
        let d = half::f16::from_bits(x_b.d).to_f32() * half::f16::from_bits(y_b.d).to_f32();

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
#[target_feature(enable = "neon")]
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
        let d0 = half::f16::from_bits(x0.d).to_f32() * half::f16::from_bits(y0.d).to_f32();
        let d1 = half::f16::from_bits(x1.d).to_f32() * half::f16::from_bits(y1.d).to_f32();

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
    }

    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);

    // Handle remainder blocks
    for ib in (chunks * 2)..nb {
        let x_b = &x[ib];
        let y_b = &y[ib];
        let d = half::f16::from_bits(x_b.d).to_f32() * half::f16::from_bits(y_b.d).to_f32();

        let isum: i32 = x_b.qs.iter().zip(y_b.qs.iter())
            .map(|(&a, &b)| a as i32 * b as i32)
            .sum();

        sumf += isum as f32 * d;
    }

    sumf
}

/// Q4_1 × Q8_1 vector dot product (NEON implementation)
#[target_feature(enable = "neon")]
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
        let m0 = half::f16::from_bits(x0.m).to_f32() * half::f16::from_bits(y0.s).to_f32();
        let m1 = half::f16::from_bits(x1.m).to_f32() * half::f16::from_bits(y1.s).to_f32();
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
        let d0 = half::f16::from_bits(x0.d).to_f32() * half::f16::from_bits(y0.d).to_f32();
        let d1 = half::f16::from_bits(x1.d).to_f32() * half::f16::from_bits(y1.d).to_f32();

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), d0);
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), d1);
    }

    let mut sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs;

    // Handle remainder blocks
    for ib in (chunks * 2)..nb {
        let x_b = &x[ib];
        let y_b = &y[ib];
        let d = half::f16::from_bits(x_b.d).to_f32() * half::f16::from_bits(y_b.d).to_f32();
        let m = half::f16::from_bits(x_b.m).to_f32() * half::f16::from_bits(y_b.s).to_f32();

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
