// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Flash Attention using online softmax algorithm

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

        if let Some(m) = mask {
            kq += m[i_kv];
            if m[i_kv] == f32::NEG_INFINITY {
                continue;
            }
        }

        let m_old = m;
        let k_vs: f32;
        let m_scale: f32;

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
