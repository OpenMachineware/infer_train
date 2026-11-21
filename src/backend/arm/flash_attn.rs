// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Flash Attention using online softmax algorithm (NEON optimized)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// FP32 Flash Attention (NEON optimized)
#[cfg(target_arch = "aarch64")]
pub fn flash_attn_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    output: &mut [f32],
    head_dim: usize,
    n_kv: usize,
    scale: f32,
) {
    assert_eq!(q.len(), head_dim);
    assert_eq!(k.len(), n_kv * head_dim);
    assert_eq!(v.len(), n_kv * head_dim);
    assert_eq!(output.len(), head_dim);
    assert!(head_dim % 4 == 0, "head_dim must be multiple of 4 for NEON");

    unsafe {
        // Initialize output to zero
        let np = head_dim & !15;
        let mut i = 0;
        while i < np {
            vst1q_f32(output.as_mut_ptr().add(i), vdupq_n_f32(0.0));
            vst1q_f32(output.as_mut_ptr().add(i + 4), vdupq_n_f32(0.0));
            vst1q_f32(output.as_mut_ptr().add(i + 8), vdupq_n_f32(0.0));
            vst1q_f32(output.as_mut_ptr().add(i + 12), vdupq_n_f32(0.0));
            i += 16;
        }
        for j in np..head_dim {
            output[j] = 0.0;
        }

        let mut m = f32::NEG_INFINITY;
        let mut s = 0.0f32;

        let scale_vec = vdupq_n_f32(scale);

        for i_kv in 0..n_kv {
            // Q·K dot product using NEON
            let k_row = k.as_ptr().add(i_kv * head_dim);
            let mut kq = 0.0f32;

            // NEON dot product
            let mut sum0 = vdupq_n_f32(0.0);
            let mut sum1 = vdupq_n_f32(0.0);
            let mut sum2 = vdupq_n_f32(0.0);
            let mut sum3 = vdupq_n_f32(0.0);

            let mut d = 0;
            while d < np {
                let qv0 = vld1q_f32(q.as_ptr().add(d));
                let kv0 = vld1q_f32(k_row.add(d));
                let qv1 = vld1q_f32(q.as_ptr().add(d + 4));
                let kv1 = vld1q_f32(k_row.add(d + 4));
                let qv2 = vld1q_f32(q.as_ptr().add(d + 8));
                let kv2 = vld1q_f32(k_row.add(d + 8));
                let qv3 = vld1q_f32(q.as_ptr().add(d + 12));
                let kv3 = vld1q_f32(k_row.add(d + 12));

                sum0 = vmlaq_f32(sum0, qv0, kv0);
                sum1 = vmlaq_f32(sum1, qv1, kv1);
                sum2 = vmlaq_f32(sum2, qv2, kv2);
                sum3 = vmlaq_f32(sum3, qv3, kv3);
                d += 16;
            }

            sum0 = vaddq_f32(sum0, sum1);
            sum2 = vaddq_f32(sum2, sum3);
            sum0 = vaddq_f32(sum0, sum2);
            kq = vaddvq_f32(sum0);

            // Handle remainder
            for j in np..head_dim {
                kq += q[j] * *k_row.add(j);
            }

            // Apply scale
            kq *= scale;

            // Apply mask
            if let Some(mask) = mask {
                kq += mask[i_kv];
                if mask[i_kv] == f32::NEG_INFINITY {
                    continue;
                }
            }

            let m_old = m;
            let m_scale: f32;
            let k_vs: f32;

            if kq > m {
                m = kq;
                m_scale = (m_old - m).exp();
                k_vs = 1.0;
            } else {
                m_scale = 1.0;
                k_vs = (kq - m).exp();
            }

            // Scale output using NEON
            let m_scale_vec = vdupq_n_f32(m_scale);
            d = 0;
            while d < np {
                let out0 = vld1q_f32(output.as_ptr().add(d));
                vst1q_f32(
                    output.as_mut_ptr().add(d),
                    vmulq_f32(out0, m_scale_vec),
                );
                let out1 = vld1q_f32(output.as_ptr().add(d + 4));
                vst1q_f32(
                    output.as_mut_ptr().add(d + 4),
                    vmulq_f32(out1, m_scale_vec),
                );
                let out2 = vld1q_f32(output.as_ptr().add(d + 8));
                vst1q_f32(
                    output.as_mut_ptr().add(d + 8),
                    vmulq_f32(out2, m_scale_vec),
                );
                let out3 = vld1q_f32(output.as_ptr().add(d + 12));
                vst1q_f32(
                    output.as_mut_ptr().add(d + 12),
                    vmulq_f32(out3, m_scale_vec),
                );
                d += 16;
            }
            for j in np..head_dim {
                output[j] *= m_scale;
            }

            // Accumulate V * k_vs using NEON
            let v_row = v.as_ptr().add(i_kv * head_dim);
            let k_vs_vec = vdupq_n_f32(k_vs);
            d = 0;
            while d < np {
                let out0 = vld1q_f32(output.as_ptr().add(d));
                let vv0 = vld1q_f32(v_row.add(d));
                vst1q_f32(
                    output.as_mut_ptr().add(d),
                    vmlaq_f32(out0, vv0, k_vs_vec),
                );
                let out1 = vld1q_f32(output.as_ptr().add(d + 4));
                let vv1 = vld1q_f32(v_row.add(d + 4));
                vst1q_f32(
                    output.as_mut_ptr().add(d + 4),
                    vmlaq_f32(out1, vv1, k_vs_vec),
                );
                let out2 = vld1q_f32(output.as_ptr().add(d + 8));
                let vv2 = vld1q_f32(v_row.add(d + 8));
                vst1q_f32(
                    output.as_mut_ptr().add(d + 8),
                    vmlaq_f32(out2, vv2, k_vs_vec),
                );
                let out3 = vld1q_f32(output.as_ptr().add(d + 12));
                let vv3 = vld1q_f32(v_row.add(d + 12));
                vst1q_f32(
                    output.as_mut_ptr().add(d + 12),
                    vmlaq_f32(out3, vv3, k_vs_vec),
                );
                d += 16;
            }
            for j in np..head_dim {
                output[j] += *v_row.add(j) * k_vs;
            }

            s = s * m_scale + k_vs;
        }

        // Final normalization
        if s > 0.0 {
            let inv_s = 1.0 / s;
            let inv_s_vec = vdupq_n_f32(inv_s);
            let mut d = 0;
            while d < np {
                let out0 = vld1q_f32(output.as_ptr().add(d));
                vst1q_f32(
                    output.as_mut_ptr().add(d),
                    vmulq_f32(out0, inv_s_vec),
                );
                let out1 = vld1q_f32(output.as_ptr().add(d + 4));
                vst1q_f32(
                    output.as_mut_ptr().add(d + 4),
                    vmulq_f32(out1, inv_s_vec),
                );
                let out2 = vld1q_f32(output.as_ptr().add(d + 8));
                vst1q_f32(
                    output.as_mut_ptr().add(d + 8),
                    vmulq_f32(out2, inv_s_vec),
                );
                let out3 = vld1q_f32(output.as_ptr().add(d + 12));
                vst1q_f32(
                    output.as_mut_ptr().add(d + 12),
                    vmulq_f32(out3, inv_s_vec),
                );
                d += 16;
            }
            for j in np..head_dim {
                output[j] *= inv_s;
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn flash_attn_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    output: &mut [f32],
    head_dim: usize,
    n_kv: usize,
    scale: f32,
) {
    assert_eq!(q.len(), head_dim);
    assert_eq!(k.len(), n_kv * head_dim);
    assert_eq!(v.len(), n_kv * head_dim);
    assert_eq!(output.len(), head_dim);

    for i in 0..head_dim {
        output[i] = 0.0;
    }

    let mut m = f32::NEG_INFINITY;
    let mut s = 0.0f32;

    for i_kv in 0..n_kv {
        let mut kq = 0.0f32;
        for d in 0..head_dim {
            kq += q[d] * k[i_kv * head_dim + d];
        }
        kq *= scale;

        if let Some(mask) = mask {
            kq += mask[i_kv];
            if mask[i_kv] == f32::NEG_INFINITY {
                continue;
            }
        }

        let m_old = m;
        let m_scale: f32;
        let k_vs: f32;

        if kq > m {
            m = kq;
            m_scale = (m_old - m).exp();
            k_vs = 1.0;
        } else {
            m_scale = 1.0;
            k_vs = (kq - m).exp();
        }

        for d in 0..head_dim {
            output[d] *= m_scale;
        }

        for d in 0..head_dim {
            output[d] += v[i_kv * head_dim + d] * k_vs;
        }

        s = s * m_scale + k_vs;
    }

    if s > 0.0 {
        let inv_s = 1.0 / s;
        for d in 0..head_dim {
            output[d] *= inv_s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_attn_basic() {
        let head_dim = 4;
        let n_kv = 3;

        let q = vec![1.0f32, 0.0, 0.0, 0.0];

        let k =
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let v =
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let mut output = vec![0.0f32; head_dim];

        flash_attn_f32(&q, &k, &v, None, &mut output, head_dim, n_kv, 1.0);

        assert!(output.iter().all(|&v| v > 0.0));
        assert!((output[0] - 3.54).abs() < 0.1);
    }
}
