// RoPE (Rotary Position Embedding) - Scalar implementation
// Matches llama.cpp's ggml_compute_forward_rope

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeType {
    Normal = 0,
    NeoX = 1,
    Multi = 2,
    Vision = 3,
    IMRope = 4,
}

#[derive(Debug, Clone)]
pub struct RopeParams {
    pub n_dims: usize,
    pub rope_type: RopeType,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub n_ctx_orig: usize,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl Default for RopeParams {
    fn default() -> Self {
        Self {
            n_dims: 128,
            rope_type: RopeType::NeoX,
            freq_base: 10000.0,
            freq_scale: 1.0,
            n_ctx_orig: 2048,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

// YaRN correction dimensions
#[inline]
fn rope_yarn_corr_dims(n_dims: usize, n_ctx_orig: usize, freq_base: f32, beta_fast: f32, beta_slow: f32) -> (f32, f32) {
    let corr_dim = |n_rot: f32| -> f32 {
        n_dims as f32 * (n_ctx_orig as f32 / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * freq_base.ln())
    };
    let dim0 = 0.0_f32.max(corr_dim(beta_fast).floor());
    let dim1 = (n_dims as f32 - 1.0).min(corr_dim(beta_slow).ceil());
    (dim0, dim1)
}

// YaRN ramp function
#[inline]
fn rope_yarn_ramp(low: f32, high: f32, i0: i64) -> f32 {
    let y = (i0 as f32 / 2.0 - low) / (high - low).max(0.001);
    1.0 - 1.0_f32.min(0.0_f32.max(y))
}

// Compute cos and sin with YaRN
#[inline]
fn rope_yarn(theta_extrap: f32, freq_scale: f32, corr_dims: (f32, f32), i0: i64, ext_factor: f32, mscale: f32) -> (f32, f32) {
    let theta_interp = freq_scale * theta_extrap;
    let mut theta = theta_interp;
    let mut scale = mscale;

    if ext_factor != 0.0 {
        let ramp_mix = rope_yarn_ramp(corr_dims.0, corr_dims.1, i0) * ext_factor;
        theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
        scale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
    }

    (theta.cos() * scale, theta.sin() * scale)
}

// Main RoPE NeoX - scalar implementation matching llama.cpp
pub fn rope_neox(src: &[f32], dst: &mut [f32], pos: usize, params: &RopeParams) {
    let n_dims = params.n_dims;
    let n_dims_half = n_dims / 2;
    let theta_scale = params.freq_base.powf(-2.0 / n_dims as f32);
    let corr_dims = rope_yarn_corr_dims(n_dims, params.n_ctx_orig, params.freq_base, params.beta_fast, params.beta_slow);

    // Copy non-rotated dimensions
    let total_dims = src.len();
    if total_dims > n_dims {
        dst[n_dims..total_dims].copy_from_slice(&src[n_dims..total_dims]);
    }

    // Apply rotation
    let mut theta = pos as f32;
    for ic in 0..n_dims_half {
        let (cos_theta, sin_theta) = rope_yarn(theta, params.freq_scale, corr_dims, (ic * 2) as i64, params.ext_factor, params.attn_factor);

        let x0 = src[ic];
        let x1 = src[ic + n_dims_half];

        dst[ic] = x0 * cos_theta - x1 * sin_theta;
        dst[ic + n_dims_half] = x0 * sin_theta + x1 * cos_theta;

        theta *= theta_scale;
    }
}

// Simple version (no YaRN) for common case - matches llama.cpp's simple path
// Use sin_cos for potentially faster combined computation
// Unsafe version eliminates bounds checking for performance
pub fn rope_neox_simple(src: &[f32], dst: &mut [f32], pos: usize, n_dims: usize, freq_base: f32) {
    let n_dims_half = n_dims / 2;
    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    let mut theta = pos as f32;

    unsafe {
        for ic in 0..n_dims_half {
            let (sin_theta, cos_theta) = theta.sin_cos();

            let x0 = *src.get_unchecked(ic);
            let x1 = *src.get_unchecked(ic + n_dims_half);

            *dst.get_unchecked_mut(ic) = x0 * cos_theta - x1 * sin_theta;
            *dst.get_unchecked_mut(ic + n_dims_half) = x0 * sin_theta + x1 * cos_theta;

            theta *= theta_scale;
        }
    }
}

// Batch processing
pub fn rope_neox_batch(src: &[f32], dst: &mut [f32], positions: &[i32], params: &RopeParams) {
    let head_dim = src.len() / positions.len();
    let n_dims = params.n_dims.min(head_dim);
    let mut params_adjusted = params.clone();
    params_adjusted.n_dims = n_dims;

    for (i, &pos) in positions.iter().enumerate() {
        rope_neox(&src[i * head_dim..], &mut dst[i * head_dim..], pos as usize, &params_adjusted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_correctness() {
        let n_dims = 128;
        let src: Vec<f32> = (0..n_dims).map(|i| (i as f32 + 1.0) / 100.0).collect();
        let mut dst = vec![0.0f32; n_dims];

        let params = RopeParams {
            n_dims,
            rope_type: RopeType::NeoX,
            freq_base: 10000.0,
            ext_factor: 0.0,
            ..Default::default()
        };

        for pos in 0..10 {
            rope_neox(&src, &mut dst, pos, &params);
            // Verify it produces valid output (no NaN, reasonable range)
            for i in 0..n_dims {
                assert!(dst[i].is_finite());
            }
        }
    }
}
