// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// RMS Normalization operator

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// FP32 RMS Norm (NEON optimized)
#[cfg(target_arch = "aarch64")]
pub fn rms_norm_f32(x: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = x.len();
    assert_eq!(n, weight.len());
    assert_eq!(n, output.len());
    assert_eq!(n % 4, 0);

    unsafe {
        let np = n & !31;
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let mut i = 0;
        while i < np {
            let v0 = vld1q_f32(x.as_ptr().add(i));
            let v1 = vld1q_f32(x.as_ptr().add(i + 4));
            let v2 = vld1q_f32(x.as_ptr().add(i + 8));
            let v3 = vld1q_f32(x.as_ptr().add(i + 12));
            let v4 = vld1q_f32(x.as_ptr().add(i + 16));
            let v5 = vld1q_f32(x.as_ptr().add(i + 20));
            let v6 = vld1q_f32(x.as_ptr().add(i + 24));
            let v7 = vld1q_f32(x.as_ptr().add(i + 28));

            sum0 = vmlaq_f32(sum0, v0, v0);
            sum1 = vmlaq_f32(sum1, v1, v1);
            sum2 = vmlaq_f32(sum2, v2, v2);
            sum3 = vmlaq_f32(sum3, v3, v3);
            sum0 = vmlaq_f32(sum0, v4, v4);
            sum1 = vmlaq_f32(sum1, v5, v5);
            sum2 = vmlaq_f32(sum2, v6, v6);
            sum3 = vmlaq_f32(sum3, v7, v7);
            i += 32;
        }

        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);

        let mut sum = vaddvq_f32(sum0);

        for j in np..n {
            sum += x[j] * x[j];
        }

        let mean = sum / (n as f32);
        let scale = 1.0 / (mean + eps).sqrt();

        i = 0;
        let scale_vec = vdupq_n_f32(scale);
        while i + 4 <= n {
            let xv = vld1q_f32(x.as_ptr().add(i));
            let wv = vld1q_f32(weight.as_ptr().add(i));

            let yv = vmulq_f32(vmulq_f32(xv, scale_vec), wv);
            vst1q_f32(output.as_mut_ptr().add(i), yv);
            i += 4;
        }

        for j in i..n {
            output[j] = x[j] * scale * weight[j];
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn rms_norm_f32(x: &[f32], weight: &[f32], output: &mut [f32], eps: f32) {
    let n = x.len();
    assert_eq!(n, weight.len());
    assert_eq!(n, output.len());

    let sum: f32 = x.iter().map(|&v| v * v).sum();
    let mean = sum / (n as f32);
    let scale = 1.0 / (mean + eps).sqrt();

    for i in 0..n {
        output[i] = x[i] * scale * weight[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight = vec![1.0f32; 8];
        let mut output = vec![0.0f32; 8];

        rms_norm_f32(&x, &weight, &mut output, 1e-5);

        let output_rms =
            (output.iter().map(|&v| v * v).sum::<f32>() / 8.0).sqrt();
        assert!((output_rms - 1.0).abs() < 0.1);
    }
}
