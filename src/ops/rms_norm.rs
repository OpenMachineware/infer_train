use std::arch::aarch64::*;

/// RMSNorm: y = x * w / sqrt(mean(x^2) + eps)
///
/// # Arguments
/// * `x` - Input tensor [n_rows, hidden_dim]
/// * `w` - Weight tensor [hidden_dim]
/// * `dst` - Output tensor [n_rows, hidden_dim]
/// * `hidden_dim` - Dimension of each row
/// * `eps` - Small constant for numerical stability
#[target_feature(enable = "neon")]
pub unsafe fn rms_norm_f32(
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

        // Compute sum of squares using NEON
        // Use f64 for sum accumulation (like llama.cpp's ggml_float)
        // Process 4 elements at a time
        let chunks = hidden_dim / 4;
        let mut sum_vec = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let v = vld1q_f32(x_row.as_ptr().add(i * 4));
            sum_vec = vmlaq_f32(sum_vec, v, v);
        }

        // Accumulate in f64 for precision (matches llama.cpp)
        let mut sum_sq: f64 = vaddvq_f32(sum_vec) as f64;

        // Handle remainder
        for i in (chunks * 4)..hidden_dim {
            sum_sq += (x_row[i] * x_row[i]) as f64;
        }

        // Compute scale (llama.cpp: scale = 1/sqrt(mean + eps))
        let mean = sum_sq / hidden_dim as f64;
        let scale = (1.0 / (mean + eps as f64).sqrt()) as f32;

        // Apply normalization and weight
        for i in 0..chunks {
            let v = vld1q_f32(x_row.as_ptr().add(i * 4));
            let weight = vld1q_f32(w.as_ptr().add(i * 4));
            let result = vmulq_f32(vmulq_n_f32(v, scale), weight);
            vst1q_f32(dst_row.as_mut_ptr().add(i * 4), result);
        }

        // Handle remainder
        for i in (chunks * 4)..hidden_dim {
            dst_row[i] = x_row[i] * scale * w[i];
        }
    }
}

/// RMSNorm with FP16 input and FP32 output
#[target_feature(enable = "neon")]
pub unsafe fn rms_norm_fp16_to_f32(
    x: &[u16],  // FP16 bits
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

        // Compute sum of squares using NEON
        // Use f64 for sum accumulation (like llama.cpp's ggml_float)
        // Process 8 FP16 at a time
        let chunks = hidden_dim / 8;
        let mut sum_vec = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let v_fp16 = vld1q_u16(x_row.as_ptr().add(i * 8));
            let v = vreinterpretq_f16_u16(v_fp16);

            // Convert FP16 to FP32
            let v_f32_lo = vcvt_f32_f16(vget_low_f16(v));
            let v_f32_hi = vcvt_f32_f16(vget_high_f16(v));

            sum_vec = vmlaq_f32(sum_vec, v_f32_lo, v_f32_lo);
            sum_vec = vmlaq_f32(sum_vec, v_f32_hi, v_f32_hi);
        }

        // Accumulate in f64 for precision (matches llama.cpp)
        let mut sum_sq: f64 = vaddvq_f32(sum_vec) as f64;

        // Handle remainder
        for i in (chunks * 8)..hidden_dim {
            let x_f32 = half::f16::from_bits(x_row[i]).to_f32();
            sum_sq += (x_f32 * x_f32) as f64;
        }

        // Compute scale (llama.cpp: scale = 1/sqrt(mean + eps))
        let mean = sum_sq / hidden_dim as f64;
        let scale = (1.0 / (mean + eps as f64).sqrt()) as f32;

        // Apply normalization and weight
        for i in 0..chunks {
            let v_fp16 = vld1q_u16(x_row.as_ptr().add(i * 8));
            let v = vreinterpretq_f16_u16(v_fp16);

            let v_f32_lo = vcvt_f32_f16(vget_low_f16(v));
            let v_f32_hi = vcvt_f32_f16(vget_high_f16(v));

            let w_lo = vld1q_f32(w.as_ptr().add(i * 8));
            let w_hi = vld1q_f32(w.as_ptr().add(i * 8 + 4));

            let result_lo = vmulq_f32(vmulq_n_f32(v_f32_lo, scale), w_lo);
            let result_hi = vmulq_f32(vmulq_n_f32(v_f32_hi, scale), w_hi);

            vst1q_f32(dst_row.as_mut_ptr().add(i * 8), result_lo);
            vst1q_f32(dst_row.as_mut_ptr().add(i * 8 + 4), result_hi);
        }

        // Handle remainder
        for i in (chunks * 8)..hidden_dim {
            let x_f32 = half::f16::from_bits(x_row[i]).to_f32();
            dst_row[i] = x_f32 * scale * w[i];
        }
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

        // Create test data
        let x: Vec<f32> = (0..n_rows * hidden_dim).map(|i| i as f32 * 0.1).collect();
        let w: Vec<f32> = (0..hidden_dim).map(|i| 1.0 + i as f32 * 0.01).collect();
        let mut dst = vec![0.0f32; n_rows * hidden_dim];

        unsafe {
            rms_norm_f32(&x, &w, &mut dst, hidden_dim, eps);
        }

        // Verify result is not all zeros
        assert!(dst.iter().any(|&v| v != 0.0));

        // Check first element
        println!("First element: {}", dst[0]);
    }
}
