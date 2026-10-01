//! Other operators: NORM, GROUP_NORM, CONCAT, REPEAT, CLAMP

use std::arch::aarch64::*;

/// Layer Normalization (standard, not RMS)
pub fn layer_norm_f32(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    y: &mut [f32],
    n: usize,
) {
    let eps = 1e-5f32;

    // Compute mean once
    let mean: f32 = x.iter().sum::<f32>() / n as f32;

    // Compute variance once
    let var: f32 =
        x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f32>() / n as f32;

    // Normalize all elements
    let std = (var + eps).sqrt();
    for i in 0..n {
        y[i] = (x[i] - mean) / std * weight[i] + bias[i];
    }
}

/// Layer Normalization with NEON
#[target_feature(enable = "neon")]
pub unsafe fn layer_norm_f32_neon(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    y: &mut [f32],
    n: usize,
) {
    let eps = 1e-5f32;

    // Compute mean using NEON
    let mut sum = vdupq_n_f32(0.0);
    let n4 = n - n % 4;
    for i in (0..n4).step_by(4) {
        sum = vaddq_f32(sum, vld1q_f32(x.as_ptr().add(i)));
    }
    let mut mean = vaddvq_f32(sum);
    for i in n4..n {
        mean += x[i];
    }
    mean /= n as f32;

    // Compute variance
    let mean_vec = vdupq_n_f32(mean);
    let mut var_sum = vdupq_n_f32(0.0);
    for i in (0..n4).step_by(4) {
        let v = vld1q_f32(x.as_ptr().add(i));
        let d = vsubq_f32(v, mean_vec);
        var_sum = vfmaq_f32(var_sum, d, d);
    }
    let mut var = vaddvq_f32(var_sum);
    for i in n4..n {
        let d = x[i] - mean;
        var += d * d;
    }
    var /= n as f32;

    let std = (var + eps).sqrt();
    let scale = 1.0 / std;
    let scale_vec = vdupq_n_f32(scale);
    let mean_vec = vdupq_n_f32(-mean * scale);

    // Normalize
    for i in (0..n4).step_by(4) {
        let x_v = vld1q_f32(x.as_ptr().add(i));
        let w_v = vld1q_f32(weight.as_ptr().add(i));
        let b_v = vld1q_f32(bias.as_ptr().add(i));

        // y = ((x - mean) / std) * w + b
        let norm = vaddq_f32(vmulq_f32(x_v, scale_vec), mean_vec);
        let y_v = vaddq_f32(vmulq_f32(norm, w_v), b_v);
        vst1q_f32(y.as_mut_ptr().add(i), y_v);
    }
    for i in n4..n {
        y[i] = (x[i] - mean) * scale * weight[i] + bias[i];
    }
}

/// Group Normalization
/// Divides channels into groups and normalizes within each group
pub fn group_norm_f32(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    y: &mut [f32],
    n_groups: usize,
    n_channels: usize,
    hw: usize,
) {
    let group_size = n_channels / n_groups;
    let eps = 1e-5f32;

    for g in 0..n_groups {
        let group_offset = g * group_size * hw;
        let group_len = group_size * hw;

        let group_slice = &x[group_offset..group_offset + group_len];
        let mean: f32 = group_slice.iter().sum::<f32>() / group_len as f32;
        let var: f32 =
            group_slice.iter().map(|&xi| (xi - mean).powi(2)).sum::<f32>()
                / group_len as f32;

        let std = (var + eps).sqrt();
        let scale = 1.0 / std;

        for c in 0..group_size {
            let ch = g * group_size + c;
            let w = weight[ch];
            let b = bias[ch];
            for s in 0..hw {
                let idx = ch * hw + s;
                y[idx] = (x[idx] - mean) * scale * w + b;
            }
        }
    }
}

/// Group Normalization with NEON
#[target_feature(enable = "neon")]
pub unsafe fn group_norm_f32_neon(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    y: &mut [f32],
    n_groups: usize,
    n_channels: usize,
    hw: usize,
) {
    let group_size = n_channels / n_groups;
    let eps = 1e-5f32;

    for g in 0..n_groups {
        // Compute mean for this group
        let mut mean = 0.0f32;
        for c in 0..group_size {
            let ch = g * group_size + c;
            let offset = ch * hw;
            let n4 = hw - hw % 4;
            let mut sum = vdupq_n_f32(0.0);
            for s in (0..n4).step_by(4) {
                sum = vaddq_f32(sum, vld1q_f32(x.as_ptr().add(offset + s)));
            }
            mean += vaddvq_f32(sum);
            for s in n4..hw {
                mean += x[offset + s];
            }
        }
        mean /= (group_size * hw) as f32;

        // Compute variance
        let mean_vec = vdupq_n_f32(mean);
        let mut var = 0.0f32;
        for c in 0..group_size {
            let ch = g * group_size + c;
            let offset = ch * hw;
            let n4 = hw - hw % 4;
            let mut var_sum = vdupq_n_f32(0.0);
            for s in (0..n4).step_by(4) {
                let v = vld1q_f32(x.as_ptr().add(offset + s));
                let d = vsubq_f32(v, mean_vec);
                var_sum = vfmaq_f32(var_sum, d, d);
            }
            var += vaddvq_f32(var_sum);
            for s in n4..hw {
                let d = x[offset + s] - mean;
                var += d * d;
            }
        }
        var /= (group_size * hw) as f32;

        let std = (var + eps).sqrt();
        let scale = 1.0 / std;
        let scale_vec = vdupq_n_f32(scale);
        let neg_mean_scale = -mean * scale;

        // Normalize each channel
        for c in 0..group_size {
            let ch = g * group_size + c;
            let w = weight[ch];
            let b = bias[ch];
            let offset = ch * hw;

            let w_vec = vdupq_n_f32(w);
            let b_vec = vdupq_n_f32(b);

            let n4 = hw - hw % 4;
            for s in (0..n4).step_by(4) {
                let x_v = vld1q_f32(x.as_ptr().add(offset + s));
                // norm = (x - mean) * scale
                let norm = vaddq_f32(
                    vmulq_f32(x_v, scale_vec),
                    vdupq_n_f32(neg_mean_scale),
                );
                // y = norm * w + b
                let y_v = vaddq_f32(vmulq_f32(norm, w_vec), b_vec);
                vst1q_f32(y.as_mut_ptr().add(offset + s), y_v);
            }
            for s in n4..hw {
                y[offset + s] = (x[offset + s] - mean) * scale * w + b;
            }
        }
    }
}

/// Concatenate tensors along a dimension
pub fn concat_f32(
    inputs: &[&[f32]],
    output: &mut [f32],
    n_tensors: usize,
    len: usize,
) {
    let mut offset = 0;
    for i in 0..n_tensors {
        output[offset..offset + len].copy_from_slice(inputs[i]);
        offset += len;
    }
}

/// Repeat tensor along a dimension
pub fn repeat_f32(
    input: &[f32],
    output: &mut [f32],
    n_repeat: usize,
    len: usize,
) {
    for i in 0..n_repeat {
        output[i * len..(i + 1) * len].copy_from_slice(input);
    }
}

/// Clamp values to a range with NEON
#[target_feature(enable = "neon")]
pub unsafe fn clamp_f32_neon(
    input: &[f32],
    output: &mut [f32],
    min: f32,
    max: f32,
) {
    let n = input.len();
    let n4 = n - n % 4;
    let min_vec = vdupq_n_f32(min);
    let max_vec = vdupq_n_f32(max);

    for i in (0..n4).step_by(4) {
        let v = vld1q_f32(input.as_ptr().add(i));
        let clamped = vminq_f32(vmaxq_f32(v, min_vec), max_vec);
        vst1q_f32(output.as_mut_ptr().add(i), clamped);
    }
    for i in n4..n {
        output[i] = if input[i] < min {
            min
        } else if input[i] > max {
            max
        } else {
            input[i]
        };
    }
}

/// Clamp values to a range
pub fn clamp_f32(input: &[f32], output: &mut [f32], min: f32, max: f32) {
    for (o, &i) in output.iter_mut().zip(input.iter()) {
        *o = if i < min {
            min
        } else if i > max {
            max
        } else {
            i
        };
    }
}

/// Fill tensor with a value
pub fn fill_f32(output: &mut [f32], value: f32) {
    for o in output.iter_mut() {
        *o = value;
    }
}

/// Pad tensor with zeros
pub fn pad_f32(
    input: &[f32],
    output: &mut [f32],
    h: usize,
    w: usize,
    pad_h: usize,
    pad_w: usize,
) {
    let h_out = h + 2 * pad_h;
    let w_out = w + 2 * pad_w;

    output.fill(0.0);

    for i in 0..h {
        let src_row = i * w;
        let dst_row = (i + pad_h) * w_out + pad_w;
        output[dst_row..dst_row + w]
            .copy_from_slice(&input[src_row..src_row + w]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0, 1.0, 1.0, 1.0];
        let b = [0.0, 0.0, 0.0, 0.0];
        let mut y = [0.0f32; 4];

        layer_norm_f32(&x, &w, &b, &mut y, 4);

        let sum: f32 = y.iter().sum();
        assert!(sum.abs() < 1e-4);
    }

    #[test]
    fn test_clamp() {
        let input = [0.0, 0.5, 1.0, 1.5, 2.0];
        let mut output = [0.0f32; 5];

        clamp_f32(&input, &mut output, 0.25, 1.25);

        assert_eq!(output, [0.25, 0.5, 1.0, 1.25, 1.25]);
    }

    #[test]
    fn test_repeat() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0f32; 6];

        repeat_f32(&input, &mut output, 2, 3);

        assert_eq!(output, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }
}
