use std::arch::aarch64::*;

/// Convert FP16 to FP32 using NEON fcvt instruction
#[inline(always)]
pub unsafe fn fp16_to_fp32(bits: u16) -> f32 {
    // Use NEON fcvt instruction for efficient conversion
    let f16_vec = vdup_n_u16(bits);
    let f16 = vreinterpret_f16_u16(f16_vec);
    let f32_vec = vcvt_f32_f16(f16);
    vgetq_lane_f32(f32_vec, 0)
}

/// FP32 vector dot product (NEON implementation)
#[target_feature(enable = "neon,dotprod")]
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
#[target_feature(enable = "neon,dotprod")]
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
        let a_f32 = fp16_to_fp32(a[i]);
        let b_f32 = fp16_to_fp32(b[i]);
        remainder += a_f32 * b_f32;
    }

    result + remainder
}

/// BF16 vector dot product (NEON implementation)
/// Note: Requires ARMv8.2-A with BF16 extension
#[cfg(target_feature = "bf16")]
#[target_feature(enable = "neon,dotprod")]
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
#[target_feature(enable = "neon,dotprod")]
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
