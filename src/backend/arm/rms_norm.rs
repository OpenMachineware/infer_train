// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// RMS Normalization and QK Normalization operators

use crate::types::QuantType;

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

/// RMS Norm with weight data in F16/F32 format
/// Returns error if weight format is unsupported
pub fn rms_norm(
    input: &[f32],
    weight_data: &[u8],
    weight_type: QuantType,
    output: &mut [f32],
    eps: f32,
) -> Result<(), String> {
    let size = input.len();
    if output.len() != size {
        return Err("Output size mismatch".into());
    }

    match weight_type {
        QuantType::F32 => {
            if weight_data.len() != size * 4 {
                return Err("F32 weight size mismatch".into());
            }
            let weight: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    weight_data.as_ptr() as *const f32,
                    size,
                )
            };
            rms_norm_f32(input, weight, output, eps);
        }
        QuantType::F16 => {
            if weight_data.len() != size * 2 {
                return Err("F16 weight size mismatch".into());
            }
            let sum_sq: f32 = input.iter().map(|x| x * x).sum();
            let rms = (sum_sq / size as f32 + eps).sqrt();
            let scale = 1.0 / rms;
            for (i, chunk) in weight_data.chunks_exact(2).enumerate() {
                let w = half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
                output[i] = input[i] * scale * w;
            }
        }
        _ => {
            return Err(format!(
                "Unsupported weight type for rms_norm: {:?}",
                weight_type
            ));
        }
    }
    Ok(())
}

/// Batch RMS Norm: [n_tokens * hidden_size] -> [n_tokens * hidden_size]
pub fn rms_norm_batch(
    input: &[f32],
    weight_data: &[u8],
    weight_type: QuantType,
    output: &mut [f32],
    n_tokens: usize,
    eps: f32,
) -> Result<(), String> {
    let hidden_size = input.len() / n_tokens;
    if output.len() != input.len() {
        return Err("Output size mismatch".into());
    }

    // Pre-convert weight to f32 for efficiency
    let weight: Vec<f32> = match weight_type {
        QuantType::F32 if weight_data.len() == hidden_size * 4 => unsafe {
            std::slice::from_raw_parts(
                weight_data.as_ptr() as *const f32,
                hidden_size,
            )
            .to_vec()
        },
        QuantType::F16 if weight_data.len() == hidden_size * 2 => weight_data
            .chunks_exact(2)
            .map(|chunk| {
                half::f16::from_le_bytes([chunk[0], chunk[1]]).to_f32()
            })
            .collect(),
        _ => return Err("Unsupported weight type or size mismatch".into()),
    };

    for token_idx in 0..n_tokens {
        let in_offset = token_idx * hidden_size;
        let out_offset = token_idx * hidden_size;
        let input_slice = &input[in_offset..in_offset + hidden_size];
        let output_slice = &mut output[out_offset..out_offset + hidden_size];
        rms_norm_f32(input_slice, &weight, output_slice, eps);
    }
    Ok(())
}

/// QK Normalization (head-wise RMS norm)
/// input: [n_heads * head_dim]
/// weight: [head_dim] (shared across heads)
pub fn qk_norm(
    input: &[f32],
    weight_data: &[u8],
    weight_type: QuantType,
    n_heads: usize,
    head_dim: usize,
    output: &mut [f32],
    eps: f32,
) -> Result<(), String> {
    if input.len() != n_heads * head_dim || output.len() != n_heads * head_dim {
        return Err("Input/output size mismatch".into());
    }

    let weight: &[f32] = match weight_type {
        QuantType::F32 if weight_data.len() == head_dim * 4 => unsafe {
            std::slice::from_raw_parts(
                weight_data.as_ptr() as *const f32,
                head_dim,
            )
        },
        _ => return Err("QK norm only supports F32 weight".into()),
    };

    for h in 0..n_heads {
        let head_start = h * head_dim;
        let head_input = &input[head_start..head_start + head_dim];
        let head_output = &mut output[head_start..head_start + head_dim];

        // Compute RMS for this head
        let ss: f32 =
            head_input.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
        let scale = 1.0f32 / (ss + eps).sqrt();

        for d in 0..head_dim {
            head_output[d] = head_input[d] * scale * weight[d];
        }
    }
    Ok(())
}

/// Batch QK Normalization
/// input: [n_tokens * n_heads * head_dim]
pub fn qk_norm_batch(
    input: &[f32],
    weight_data: &[u8],
    weight_type: QuantType,
    n_heads: usize,
    head_dim: usize,
    n_tokens: usize,
    output: &mut [f32],
    eps: f32,
) -> Result<(), String> {
    let expected_len = n_tokens * n_heads * head_dim;
    if input.len() != expected_len || output.len() != expected_len {
        return Err("Input/output size mismatch".into());
    }

    let weight: &[f32] = match weight_type {
        QuantType::F32 if weight_data.len() == head_dim * 4 => unsafe {
            std::slice::from_raw_parts(
                weight_data.as_ptr() as *const f32,
                head_dim,
            )
        },
        _ => return Err("QK norm only supports F32 weight".into()),
    };

    for token_idx in 0..n_tokens {
        for h in 0..n_heads {
            let head_start = token_idx * n_heads * head_dim + h * head_dim;
            let head_input = &input[head_start..head_start + head_dim];
            let head_output = &mut output[head_start..head_start + head_dim];

            let ss: f32 =
                head_input.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
            let scale = 1.0f32 / (ss + eps).sqrt();

            for d in 0..head_dim {
                head_output[d] = head_input[d] * scale * weight[d];
            }
        }
    }
    Ok(())
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
