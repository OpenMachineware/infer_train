// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Rotary Position Embedding (RoPE) operator (NEON optimized)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// FP32 RoPE (NEON optimized)
#[cfg(target_arch = "aarch64")]
pub fn rope_f32(
    x: &[f32],
    output: &mut [f32],
    pos: usize,
    n_dims: usize,
    head_dim: usize,
    freq_base: f32,
) {
    let n = x.len();
    assert_eq!(n, output.len());
    assert_eq!(n_dims % 2, 0);
    assert!(head_dim % 4 == 0, "head_dim must be multiple of 4 for NEON");

    unsafe {
        // Copy input to output first
        let np = n & !15;
        let mut i = 0;
        while i < np {
            vst1q_f32(output.as_mut_ptr().add(i), vld1q_f32(x.as_ptr().add(i)));
            vst1q_f32(
                output.as_mut_ptr().add(i + 4),
                vld1q_f32(x.as_ptr().add(i + 4)),
            );
            vst1q_f32(
                output.as_mut_ptr().add(i + 8),
                vld1q_f32(x.as_ptr().add(i + 8)),
            );
            vst1q_f32(
                output.as_mut_ptr().add(i + 12),
                vld1q_f32(x.as_ptr().add(i + 12)),
            );
            i += 16;
        }
        for j in np..n {
            output[j] = x[j];
        }

        let n_heads = n / head_dim;
        let half_n_dims = n_dims / 2;

        for h in 0..n_heads {
            let offset = h * head_dim;

            // Process 4 pairs at a time using NEON
            let mut i = 0;
            while i + 3 < half_n_dims {
                let idx0 = offset + i;
                let idx1 = offset + i + half_n_dims;

                // Compute frequencies for 4 pairs
                let mut freqs = [0.0f32; 4];
                for k in 0..4 {
                    let freq_dim = (i + k) as f32;
                    freqs[k] = freq_base.powf(-2.0 * freq_dim / n_dims as f32);
                }
                let theta0 = pos as f32 * freqs[0];
                let theta1 = pos as f32 * freqs[1];
                let theta2 = pos as f32 * freqs[2];
                let theta3 = pos as f32 * freqs[3];

                let cos_sin = [
                    theta0.cos(),
                    theta0.sin(),
                    theta1.cos(),
                    theta1.sin(),
                    theta2.cos(),
                    theta2.sin(),
                    theta3.cos(),
                    theta3.sin(),
                ];

                // Load x values
                let x0 = vld1q_f32(x.as_ptr().add(idx0));
                let x1 = vld1q_f32(x.as_ptr().add(idx1));

                // Load cos and sin
                let cos_v = vld1q_f32(cos_sin.as_ptr()); // [cos0, sin0, cos1, sin1]
                let sin_v = vld1q_f32(cos_sin.as_ptr().add(4)); // [cos2, sin2, cos3, sin3]

                // Rearrange cos and sin
                // cos_vec = [cos0, cos0, cos1, cos1]
                // sin_vec = [sin0, sin0, sin1, sin1]
                let cos0 = vgetq_lane_f32(cos_v, 0);
                let cos1 = vgetq_lane_f32(cos_v, 2);
                let sin0 = vgetq_lane_f32(cos_v, 1);
                let sin1 = vgetq_lane_f32(cos_v, 3);

                let cos2 = vgetq_lane_f32(sin_v, 0);
                let cos3 = vgetq_lane_f32(sin_v, 2);
                let sin2 = vgetq_lane_f32(sin_v, 1);
                let sin3 = vgetq_lane_f32(sin_v, 3);

                // Process 2 pairs at a time
                let cos_vec_01 = vdupq_n_f32(cos0);
                let cos_vec_01b = vdupq_n_f32(cos1);
                let sin_vec_01 = vdupq_n_f32(sin0);
                let sin_vec_01b = vdupq_n_f32(sin1);

                // output[idx0] = x0 * cos - x1 * sin
                // output[idx1] = x0 * sin + x1 * cos
                let x0_01 = vld1q_f32(x.as_ptr().add(idx0));
                let x1_01 = vld1q_f32(x.as_ptr().add(idx1));

                let out0_01 = vsubq_f32(
                    vmulq_f32(x0_01, cos_vec_01),
                    vmulq_f32(x1_01, sin_vec_01),
                );
                let out1_01 = vaddq_f32(
                    vmulq_f32(x0_01, sin_vec_01),
                    vmulq_f32(x1_01, cos_vec_01),
                );

                vst1q_f32(output.as_mut_ptr().add(idx0), out0_01);
                vst1q_f32(output.as_mut_ptr().add(idx1), out1_01);

                // Next pair
                let cos_vec_23 = vdupq_n_f32(cos2);
                let cos_vec_23b = vdupq_n_f32(cos3);
                let sin_vec_23 = vdupq_n_f32(sin2);
                let sin_vec_23b = vdupq_n_f32(sin3);

                let x0_23 = vld1q_f32(x.as_ptr().add(idx0 + 4));
                let x1_23 = vld1q_f32(x.as_ptr().add(idx1 + 4));

                let out0_23 = vsubq_f32(
                    vmulq_f32(x0_23, cos_vec_23),
                    vmulq_f32(x1_23, sin_vec_23),
                );
                let out1_23 = vaddq_f32(
                    vmulq_f32(x0_23, sin_vec_23),
                    vmulq_f32(x1_23, cos_vec_23),
                );

                vst1q_f32(output.as_mut_ptr().add(idx0 + 4), out0_23);
                vst1q_f32(output.as_mut_ptr().add(idx1 + 4), out1_23);

                i += 8;
            }

            // Handle remainder
            for j in i..half_n_dims {
                let idx0 = offset + j;
                let idx1 = offset + j + half_n_dims;
                if idx1 >= n {
                    break;
                }

                let freq_dim = j as f32;
                let freq = freq_base.powf(-2.0 * freq_dim / n_dims as f32);
                let theta = pos as f32 * freq;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                let x0 = x[idx0];
                let x1 = x[idx1];

                output[idx0] = x0 * cos_theta - x1 * sin_theta;
                output[idx1] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
}

/// In-place RoPE (NEON optimized)
#[cfg(target_arch = "aarch64")]
pub fn rope_f32_inplace(
    x: &mut [f32],
    pos: usize,
    n_dims: usize,
    head_dim: usize,
    freq_base: f32,
) {
    let n = x.len();
    assert_eq!(n_dims % 2, 0);
    assert!(head_dim % 4 == 0, "head_dim must be multiple of 4 for NEON");

    unsafe {
        let n_heads = n / head_dim;
        let half_n_dims = n_dims / 2;

        for h in 0..n_heads {
            let offset = h * head_dim;

            // Process 4 pairs at a time using NEON
            let mut i = 0;
            while i + 3 < half_n_dims {
                let idx0 = offset + i;
                let idx1 = offset + i + half_n_dims;

                // Compute frequencies for 4 pairs
                let mut freqs = [0.0f32; 4];
                for k in 0..4 {
                    let freq_dim = (i + k) as f32;
                    freqs[k] = freq_base.powf(-2.0 * freq_dim / n_dims as f32);
                }
                let theta0 = pos as f32 * freqs[0];
                let theta1 = pos as f32 * freqs[1];
                let theta2 = pos as f32 * freqs[2];
                let theta3 = pos as f32 * freqs[3];

                let cos0 = theta0.cos();
                let sin0 = theta0.sin();
                let cos1 = theta1.cos();
                let sin1 = theta1.sin();
                let cos2 = theta2.cos();
                let sin2 = theta2.sin();
                let cos3 = theta3.cos();
                let sin3 = theta3.sin();

                // Process 4 pairs
                let x0 = x[idx0];
                let x1 = x[idx1];
                x[idx0] = x0 * cos0 - x1 * sin0;
                x[idx1] = x0 * sin0 + x1 * cos0;

                let x0 = x[idx0 + 1];
                let x1 = x[idx1 + 1];
                x[idx0 + 1] = x0 * cos1 - x1 * sin1;
                x[idx1 + 1] = x0 * sin1 + x1 * cos1;

                let x0 = x[idx0 + 2];
                let x1 = x[idx1 + 2];
                x[idx0 + 2] = x0 * cos2 - x1 * sin2;
                x[idx1 + 2] = x0 * sin2 + x1 * cos2;

                let x0 = x[idx0 + 3];
                let x1 = x[idx1 + 3];
                x[idx0 + 3] = x0 * cos3 - x1 * sin3;
                x[idx1 + 3] = x0 * sin3 + x1 * cos3;

                i += 4;
            }

            // Handle remainder
            for j in i..half_n_dims {
                let idx0 = offset + j;
                let idx1 = offset + j + half_n_dims;
                if idx1 >= n {
                    break;
                }

                let freq_dim = j as f32;
                let freq = freq_base.powf(-2.0 * freq_dim / n_dims as f32);
                let theta = pos as f32 * freq;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                let x0 = x[idx0];
                let x1 = x[idx1];

                x[idx0] = x0 * cos_theta - x1 * sin_theta;
                x[idx1] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn rope_f32(
    x: &[f32],
    output: &mut [f32],
    pos: usize,
    n_dims: usize,
    head_dim: usize,
    freq_base: f32,
) {
    let n = x.len();
    assert_eq!(n, output.len());
    assert_eq!(n_dims % 2, 0);

    output.copy_from_slice(x);

    let n_heads = n / head_dim;

    for h in 0..n_heads {
        let offset = h * head_dim;
        let half_n_dims = n_dims / 2;
        for i in 0..half_n_dims {
            let idx0 = offset + i;
            let idx1 = offset + i + half_n_dims;
            if idx1 >= n {
                break;
            }

            let freq_dim = i as f32;
            let freq = freq_base.powf(-2.0 * freq_dim / n_dims as f32);
            let theta = pos as f32 * freq;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            let x0 = x[idx0];
            let x1 = x[idx1];

            output[idx0] = x0 * cos_theta - x1 * sin_theta;
            output[idx1] = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn rope_f32_inplace(
    x: &mut [f32],
    pos: usize,
    n_dims: usize,
    head_dim: usize,
    freq_base: f32,
) {
    let n = x.len();
    assert_eq!(n_dims % 2, 0);

    let n_heads = n / head_dim;

    for h in 0..n_heads {
        let offset = h * head_dim;
        let half_n_dims = n_dims / 2;

        for i in 0..half_n_dims {
            let idx0 = offset + i;
            let idx1 = offset + i + half_n_dims;
            if idx1 >= n {
                break;
            }

            let freq_dim = i as f32;
            let freq = freq_base.powf(-2.0 * freq_dim / n_dims as f32);
            let theta = pos as f32 * freq;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            let x0 = x[idx0];
            let x1 = x[idx1];

            x[idx0] = x0 * cos_theta - x1 * sin_theta;
            x[idx1] = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope() {
        let head_dim = 64;
        let n_heads = 2;
        let n = head_dim * n_heads;
        let n_dims = 64;

        let x = vec![1.0f32; n];
        let mut output = vec![0.0f32; n];

        rope_f32(&x, &mut output, 0, n_dims, head_dim, 10000.0);

        for i in 0..n {
            assert!((output[i] - 1.0).abs() < 0.1);
        }

        rope_f32(&x, &mut output, 100, n_dims, head_dim, 10000.0);

        assert!((output[0] - 1.0).abs() > 0.1);
    }
}
