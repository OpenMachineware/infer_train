// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Q8_K quantization for vector dot products with K-quant weights

use crate::quant::types::{BlockQ8K, QK_K};

/// Quantize FP32 vector to Q8_K format
/// Q8_K is used as input format for vec_dot with K-quant weights
pub fn quantize_row_q8_k(x: &[f32], y: &mut [BlockQ8K]) {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;

    for i in 0..nb {
        let x_block = &x[i * QK_K..(i + 1) * QK_K];

        // Find max absolute value
        let mut amax = 0.0f32;
        let mut max_val = 0.0f32;
        for &val in x_block {
            let abs_val = val.abs();
            if abs_val > amax {
                amax = abs_val;
                max_val = val;
            }
        }

        // Handle zero case
        if amax == 0.0 {
            y[i].d = 0.0;
            y[i].qs.fill(0);
            y[i].bsums.fill(0);
            continue;
        }

        // Calculate scale
        // Use -127 instead of -128 for better accuracy (matches llama.cpp)
        let iscale = -127.0 / max_val;
        let scale = 1.0 / iscale;

        // Quantize values
        for j in 0..QK_K {
            let v = (x_block[j] * iscale).round() as i32;
            y[i].qs[j] = v.min(127).max(-128) as i8;
        }

        // Compute block sums (needed for K-quant dot products)
        for j in 0..16 {
            let mut sum = 0i32;
            for k in 0..16 {
                sum += y[i].qs[j * 16 + k] as i32;
            }
            y[i].bsums[j] = sum as i16;
        }

        y[i].d = scale;
    }
}

/// Quantize FP32 to Q8_K with zero-padding for non-multiple-of-QK_K sizes
/// Returns the padded input vector (owned) and mutable output slice
/// For performance, the caller should cache the padded buffer
pub fn quantize_row_q8_k_padded<'a>(
    x: &[f32],
    y: &'a mut [BlockQ8K],
    padded_x: &mut Vec<f32>,
) -> &'a [BlockQ8K] {
    let n = x.len();
    let n_padded = ((n + QK_K - 1) / QK_K) * QK_K;
    let nb = n_padded / QK_K;

    // Pad input with zeros if needed
    if n != n_padded {
        padded_x.clear();
        padded_x.reserve(n_padded);
        padded_x.extend_from_slice(x);
        padded_x.resize(n_padded, 0.0);
        quantize_row_q8_k_neon(padded_x, y);
    } else {
        quantize_row_q8_k_neon(x, y);
    }

    &y[..nb]
}

/// Quantize FP32 to Q8_K (NEON optimized version)
#[cfg(target_arch = "aarch64")]
pub fn quantize_row_q8_k_neon(x: &[f32], y: &mut [BlockQ8K]) {
    use std::arch::aarch64::*;

    // Support non-multiple-of-QK_K sizes (zero-padding is caller's responsibility)
    let n = x.len();
    let nb = (n + QK_K - 1) / QK_K; // Round up
    assert!(y.len() >= nb, "Output buffer too small: {} < {}", y.len(), nb);

    unsafe {
        for i in 0..nb {
            let x_ptr = x.as_ptr().add(i * QK_K);

            // Load all 256 values (32 x float32x4_t)
            let mut max_vals = vdupq_n_f32(0.0);

            // Find max absolute value across all 256 elements
            for j in 0..64 {
                let v = vld1q_f32(x_ptr.add(j * 4));
                max_vals = vmaxq_f32(max_vals, vabsq_f32(v));
            }

            // Reduce to single max
            let _max_dup = vdupq_n_f32(vmaxvq_f32(max_vals));
            let amax = vmaxvq_f32(max_vals);

            if amax == 0.0 {
                y[i].d = 0.0;
                y[i].qs.fill(0);
                y[i].bsums.fill(0);
                continue;
            }

            // Calculate scale
            let iscale = -127.0 / amax;
            let scale = 1.0 / iscale;
            y[i].d = scale;

            let iscale_vec = vdupq_n_f32(iscale);

            // Quantize all 256 values
            for j in 0..64 {
                let v = vld1q_f32(x_ptr.add(j * 4));
                let scaled = vmulq_f32(v, iscale_vec);
                let rounded = vrndnq_f32(scaled);
                let q = vcvtq_s32_f32(rounded);

                // Clamp to [-128, 127]
                // Note: NEON doesn't have a direct clamp, so we use min/max
                let q_clamped = vminq_s32(
                    vmaxq_s32(q, vdupq_n_s32(-128)),
                    vdupq_n_s32(127),
                );

                // Store as i8 (need to convert i32 to i8)
                y[i].qs[j * 4 + 0] = vgetq_lane_s32(q_clamped, 0) as i8;
                y[i].qs[j * 4 + 1] = vgetq_lane_s32(q_clamped, 1) as i8;
                y[i].qs[j * 4 + 2] = vgetq_lane_s32(q_clamped, 2) as i8;
                y[i].qs[j * 4 + 3] = vgetq_lane_s32(q_clamped, 3) as i8;
            }

            // Compute block sums using NEON
            for j in 0..16 {
                let qs_ptr = y[i].qs.as_ptr().add(j * 16);

                // Load 16 i8 values
                let q8 = vld1q_s8(qs_ptr);

                // Extend to i16 and pairwise add
                let s0 =
                    vaddq_s16(vmovl_s8(vget_low_s8(q8)), vmovl_high_s8(q8));

                // Reduce to single i16
                let sum = vaddvq_s16(s0);
                y[i].bsums[j] = sum;
            }
        }
    }
}
