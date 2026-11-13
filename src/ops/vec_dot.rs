// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Vector dot product operators (the most fundamental operator)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::quant::*;

// FP32 dot product (NEON optimized)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_f32_neon(x: &[f32], y: &[f32]) -> f32 {
    unsafe {
        let n = x.len();
        assert_eq!(n, y.len());
        assert_eq!(n % 4, 0);

        let np = n & !31; // Process 32 elements at a time (8 NEON registers)

        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);
        let mut sum4 = vdupq_n_f32(0.0);
        let mut sum5 = vdupq_n_f32(0.0);
        let mut sum6 = vdupq_n_f32(0.0);
        let mut sum7 = vdupq_n_f32(0.0);

        let mut i = 0;
        while i < np {
            sum0 = vmlaq_f32(
                sum0,
                vld1q_f32(x.as_ptr().add(i)),
                vld1q_f32(y.as_ptr().add(i)),
            );
            sum1 = vmlaq_f32(
                sum1,
                vld1q_f32(x.as_ptr().add(i + 4)),
                vld1q_f32(y.as_ptr().add(i + 4)),
            );
            sum2 = vmlaq_f32(
                sum2,
                vld1q_f32(x.as_ptr().add(i + 8)),
                vld1q_f32(y.as_ptr().add(i + 8)),
            );
            sum3 = vmlaq_f32(
                sum3,
                vld1q_f32(x.as_ptr().add(i + 12)),
                vld1q_f32(y.as_ptr().add(i + 12)),
            );
            sum4 = vmlaq_f32(
                sum4,
                vld1q_f32(x.as_ptr().add(i + 16)),
                vld1q_f32(y.as_ptr().add(i + 16)),
            );
            sum5 = vmlaq_f32(
                sum5,
                vld1q_f32(x.as_ptr().add(i + 20)),
                vld1q_f32(y.as_ptr().add(i + 20)),
            );
            sum6 = vmlaq_f32(
                sum6,
                vld1q_f32(x.as_ptr().add(i + 24)),
                vld1q_f32(y.as_ptr().add(i + 24)),
            );
            sum7 = vmlaq_f32(
                sum7,
                vld1q_f32(x.as_ptr().add(i + 28)),
                vld1q_f32(y.as_ptr().add(i + 28)),
            );
            i += 32;
        }

        // Reduce sums
        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum4 = vaddq_f32(sum4, sum5);
        sum6 = vaddq_f32(sum6, sum7);
        sum0 = vaddq_f32(sum0, sum2);
        sum4 = vaddq_f32(sum4, sum6);
        sum0 = vaddq_f32(sum0, sum4);

        // Horizontal sum
        let mut result = vaddvq_f32(sum0);

        // Handle leftovers
        for j in np..n {
            result += x[j] * y[j];
        }

        result
    }
}

// Q4_0 x Q8_0 dot product (NEON optimized)
// This is the core quantized GEMV kernel
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_q4_0_q8_0_neon(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    unsafe {
        let nb = x.len();
        assert_eq!(nb, y.len());

        let m4b = vdupq_n_u8(0x0F);
        let s8b = vdupq_n_s8(0x8);

        let mut sumv = vdupq_n_f32(0.0);

        for i in 0..nb {
            let block_x = &x[i];
            let block_y = &y[i];

            // Load 4-bit values and expand to 8-bit
            let v = vld1q_u8(block_x.qs.as_ptr());
            let vxl = vandq_u8(v, m4b);
            let vxh = vshrq_n_u8(v, 4);

            // Subtract 8 to make unsigned 4-bit into signed
            let vxl = vsubq_s8(vreinterpretq_s8_u8(vxl), s8b);
            let vxh = vsubq_s8(vreinterpretq_s8_u8(vxh), s8b);

            // Load 8-bit values
            let vy = vld1q_s8(block_y.qs.as_ptr());

            // Low 16 bytes
            let p0 = vmull_s8(vget_low_s8(vxl), vget_low_s8(vy));
            let p1 = vmull_high_s8(vxl, vy);

            // High 16 bytes
            let vy2 = vld1q_s8(block_y.qs.as_ptr().add(16));
            let p2 = vmull_s8(vget_low_s8(vxh), vget_low_s8(vy2));
            let p3 = vmull_high_s8(vxh, vy2);

            // Pairwise add to reduce 16x16 to 4x32
            let sumb = vaddq_s32(
                vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)),
                vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3)),
            );

            // Scale by product of scales
            let dx = fp16_to_fp32(block_x.d);
            let dy = fp16_to_fp32(block_y.d);
            let scale = dx * dy;

            // Accumulate
            sumv = vmlaq_n_f32(sumv, vcvtq_f32_s32(sumb), scale);
        }

        vaddvq_f32(sumv)
    }
}

// Public interface
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_f32(x: &[f32], y: &[f32]) -> f32 {
    unsafe { vec_dot_f32_neon(x, y) }
}

#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    unsafe { vec_dot_q4_0_q8_0_neon(x, y) }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_dot_f32(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_dot_f32() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = vec_dot_f32(&x, &y);
        assert!((result - 30.0).abs() < 1e-6);
    }
}
