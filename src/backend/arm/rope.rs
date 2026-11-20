// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Rotary Position Embedding (RoPE) operator

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

        // NEOX style: rotate pairs (i, i + n_dims/2)
        // This is the standard for modern LLMs like Qwen, LLaMA, etc.
        let half_n_dims = n_dims / 2;
        for i in 0..half_n_dims {
            let idx0 = offset + i;
            let idx1 = offset + i + half_n_dims;
            if idx1 >= n {
                break;
            }

            // Standard RoPE: freq = freq_base^(-2*i/n_dims)
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

/// In-place RoPE (for zero-allocation path)
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
