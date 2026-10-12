// RMSNorm operator implementation

#[cfg(target_arch = "aarch64")]
use super::arm::neon;

/// Scalar RMSNorm: y = x * w / sqrt(mean(x^2) + eps)
pub fn rms_norm_f32_scalar(
    x: &[f32],
    w: &[f32],
    dst: &mut [f32],
    hidden_dim: usize,
    eps: f32,
) {
    let n_rows = x.len() / hidden_dim;

    for row in 0..n_rows {
        let row_offset = row * hidden_dim;
        let x_row = &x[row_offset..row_offset + hidden_dim];
        let dst_row = &mut dst[row_offset..row_offset + hidden_dim];

        // Compute sum of squares
        let mut sum_sq: f64 = 0.0;
        for &xi in x_row {
            sum_sq += (xi * xi) as f64;
        }

        // Compute scale
        let mean = sum_sq / hidden_dim as f64;
        let scale = (1.0 / (mean + eps as f64).sqrt()) as f32;

        // Apply normalization and weight
        for i in 0..hidden_dim {
            dst_row[i] = x_row[i] * scale * w[i];
        }
    }
}

/// Auto-dispatched RMSNorm using best available implementation
pub fn rms_norm_f32(
    x: &[f32],
    w: &[f32],
    dst: &mut [f32],
    hidden_dim: usize,
    eps: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64
        unsafe {
            neon::rms_norm_f32_neon(x, w, dst, hidden_dim, eps);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        rms_norm_f32_scalar(x, w, dst, hidden_dim, eps);
    }
}

/// Scalar RMSNorm with FP16 input
pub fn rms_norm_fp16_to_f32_scalar(
    x: &[u16],
    w: &[f32],
    dst: &mut [f32],
    hidden_dim: usize,
    eps: f32,
) {
    let n_rows = x.len() / hidden_dim;

    for row in 0..n_rows {
        let row_offset = row * hidden_dim;
        let x_row = &x[row_offset..row_offset + hidden_dim];
        let dst_row = &mut dst[row_offset..row_offset + hidden_dim];

        // Compute sum of squares
        let mut sum_sq: f64 = 0.0;
        for &xi in x_row {
            let x_f32 = half::f16::from_bits(xi).to_f32();
            sum_sq += (x_f32 * x_f32) as f64;
        }

        // Compute scale
        let mean = sum_sq / hidden_dim as f64;
        let scale = (1.0 / (mean + eps as f64).sqrt()) as f32;

        // Apply normalization and weight
        for i in 0..hidden_dim {
            let x_f32 = half::f16::from_bits(x_row[i]).to_f32();
            dst_row[i] = x_f32 * scale * w[i];
        }
    }
}

/// Auto-dispatched FP16 RMSNorm
pub fn rms_norm_fp16_to_f32(
    x: &[u16],
    w: &[f32],
    dst: &mut [f32],
    hidden_dim: usize,
    eps: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            neon::rms_norm_fp16_to_f32_neon(x, w, dst, hidden_dim, eps);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        rms_norm_fp16_to_f32_scalar(x, w, dst, hidden_dim, eps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_f32() {
        let hidden_dim = 64;
        let n_rows = 4;
        let eps = 1e-5;

        let x: Vec<f32> =
            (0..n_rows * hidden_dim).map(|i| i as f32 * 0.1).collect();
        let w: Vec<f32> =
            (0..hidden_dim).map(|i| 1.0 + i as f32 * 0.01).collect();
        let mut dst = vec![0.0f32; n_rows * hidden_dim];

        rms_norm_f32(&x, &w, &mut dst, hidden_dim, eps);

        // Verify result is not all zeros
        assert!(dst.iter().any(|&v| v != 0.0));
    }
}
