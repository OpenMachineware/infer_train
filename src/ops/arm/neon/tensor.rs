// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// NEON SIMD implementations for tensor operations

use std::arch::aarch64::*;

/// Layer Normalization with NEON
#[target_feature(enable = "neon")]
pub unsafe fn layer_norm_f32(
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

/// Group Normalization with NEON
#[target_feature(enable = "neon")]
pub unsafe fn group_norm_f32(
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

/// Clamp values to a range with NEON
#[target_feature(enable = "neon")]
pub unsafe fn clamp_f32(input: &[f32], output: &mut [f32], min: f32, max: f32) {
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
