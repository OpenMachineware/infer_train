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
