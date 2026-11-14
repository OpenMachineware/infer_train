// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Quantization: F32 -> Quantized formats

use crate::types::{BlockQ8_0, BlockQ8_K, QK_K, QK8_0};

// ============================================================================
// Q8_K Quantization (for K-quant GEMV input)
// ============================================================================

/// Quantize FP32 vector to Q8_K format
/// Q8_K is used as input format for vec_dot with K-quant weights
pub fn quantize_row_q8_k(x: &[f32], y: &mut [BlockQ8_K]) {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;

    for i in 0..nb {
        let x_block = &x[i * QK_K..(i + 1) * QK_K];

        // Find max absolute value
        let mut amax = 0.0f32;
        for &val in x_block {
            let abs_val = val.abs();
            if abs_val > amax {
                amax = abs_val;
            }
        }

        // Handle zero case
        if amax == 0.0 {
            y[i].d = 0.0;
            y[i].qs.fill(0);
            y[i].bsums.fill(0);
            continue;
        }

        // Calculate scale (use -127 instead of -128 for better accuracy)
        let iscale = -127.0 / amax;
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

/// Quantize FP32 to Q8_K (NEON optimized version)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn quantize_row_q8_k_neon(x: &[f32], y: &mut [BlockQ8_K]) {
    let n = x.len();
    let nb = (n + QK_K - 1) / QK_K;

    for i in 0..nb {
        let x_ptr = unsafe { x.as_ptr().add(i * QK_K) };

        // Find max absolute value across all 256 elements
        let mut max_vals = std::arch::aarch64::vdupq_n_f32(0.0);
        for j in 0..64 {
            let v = unsafe { std::arch::aarch64::vld1q_f32(x_ptr.add(j * 4)) };
            max_vals = std::arch::aarch64::vmaxq_f32(
                max_vals,
                std::arch::aarch64::vabsq_f32(v),
            );
        }

        let amax = std::arch::aarch64::vmaxvq_f32(max_vals);

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

        let iscale_vec = std::arch::aarch64::vdupq_n_f32(iscale);

        // Quantize all 256 values
        for j in 0..64 {
            let v = unsafe { std::arch::aarch64::vld1q_f32(x_ptr.add(j * 4)) };
            let scaled = std::arch::aarch64::vmulq_f32(v, iscale_vec);
            let rounded = std::arch::aarch64::vrndnq_f32(scaled);
            let q = std::arch::aarch64::vcvtq_s32_f32(rounded);

            // Clamp to [-128, 127]
            let q_clamped = std::arch::aarch64::vminq_s32(
                std::arch::aarch64::vmaxq_s32(
                    q,
                    std::arch::aarch64::vdupq_n_s32(-128),
                ),
                std::arch::aarch64::vdupq_n_s32(127),
            );

            // Store as i8
            y[i].qs[j * 4 + 0] =
                std::arch::aarch64::vgetq_lane_s32(q_clamped, 0) as i8;
            y[i].qs[j * 4 + 1] =
                std::arch::aarch64::vgetq_lane_s32(q_clamped, 1) as i8;
            y[i].qs[j * 4 + 2] =
                std::arch::aarch64::vgetq_lane_s32(q_clamped, 2) as i8;
            y[i].qs[j * 4 + 3] =
                std::arch::aarch64::vgetq_lane_s32(q_clamped, 3) as i8;
        }

        // Compute block sums using NEON
        for j in 0..16 {
            let qs_ptr = unsafe { y[i].qs.as_ptr().add(j * 16) };
            let q8 = unsafe { std::arch::aarch64::vld1q_s8(qs_ptr) };

            // Extend to i16 and add
            let s0 = std::arch::aarch64::vaddq_s16(
                std::arch::aarch64::vmovl_s8(std::arch::aarch64::vget_low_s8(
                    q8,
                )),
                std::arch::aarch64::vmovl_high_s8(q8),
            );

            let sum = std::arch::aarch64::vaddvq_s16(s0);
            y[i].bsums[j] = sum;
        }
    }
}

// ============================================================================
// Q8_0 Quantization (for Q4_0/Q5_0 GEMV input)
// ============================================================================

/// Quantize FP32 vector to Q8_0 format
/// Q8_0 is used as input format for vec_dot with Q4_0/Q5_0 weights
pub fn quantize_row_q8_0(x: &[f32], y: &mut [BlockQ8_0]) {
    assert!(x.len() % QK8_0 == 0);
    let nb = x.len() / QK8_0;

    for i in 0..nb {
        let x_block = &x[i * QK8_0..(i + 1) * QK8_0];

        // Find max absolute value
        let mut amax = 0.0f32;
        for &val in x_block {
            let abs_val = val.abs();
            if abs_val > amax {
                amax = abs_val;
            }
        }

        // Handle zero case
        if amax == 0.0 {
            y[i].d = 0;
            y[i].qs.fill(0);
            continue;
        }

        // Calculate scale (FP16)
        let scale = amax / 127.0;

        // Quantize values
        for j in 0..QK8_0 {
            let v = (x_block[j] / scale).round() as i32;
            y[i].qs[j] = v.min(127).max(-128) as i8;
        }

        y[i].d = crate::types::fp32_to_fp16(scale);
    }
}

/// Quantize FP32 to Q8_0 (NEON optimized version)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn quantize_row_q8_0_neon(x: &[f32], y: &mut [BlockQ8_0]) {
    let n = x.len();
    let nb = n / QK8_0;

    for i in 0..nb {
        let x_ptr = unsafe { x.as_ptr().add(i * QK8_0) };

        // Find max absolute value across 32 elements
        let mut max_vals = std::arch::aarch64::vdupq_n_f32(0.0);
        for j in 0..8 {
            let v = unsafe { std::arch::aarch64::vld1q_f32(x_ptr.add(j * 4)) };
            max_vals = std::arch::aarch64::vmaxq_f32(
                max_vals,
                std::arch::aarch64::vabsq_f32(v),
            );
        }

        let amax = std::arch::aarch64::vmaxvq_f32(max_vals);

        if amax == 0.0 {
            y[i].d = 0;
            y[i].qs.fill(0);
            continue;
        }

        // Calculate scale
        let scale = amax / 127.0;
        y[i].d = crate::types::fp32_to_fp16(scale);

        let iscale = 127.0 / amax;
        let iscale_vec = std::arch::aarch64::vdupq_n_f32(iscale);

        // Quantize all 32 values
        for j in 0..8 {
            let v = unsafe { std::arch::aarch64::vld1q_f32(x_ptr.add(j * 4)) };
            let scaled = std::arch::aarch64::vmulq_f32(v, iscale_vec);
            let rounded = std::arch::aarch64::vrndnq_f32(scaled);
            let q = std::arch::aarch64::vcvtq_s32_f32(rounded);

            // Clamp to [-128, 127]
            let q_clamped = std::arch::aarch64::vminq_s32(
                std::arch::aarch64::vmaxq_s32(
                    q,
                    std::arch::aarch64::vdupq_n_s32(-128),
                ),
                std::arch::aarch64::vdupq_n_s32(127),
            );

            // Store as i8
            y[i].qs[j * 4 + 0] =
                std::arch::aarch64::vgetq_lane_s32(q_clamped, 0) as i8;
            y[i].qs[j * 4 + 1] =
                std::arch::aarch64::vgetq_lane_s32(q_clamped, 1) as i8;
            y[i].qs[j * 4 + 2] =
                std::arch::aarch64::vgetq_lane_s32(q_clamped, 2) as i8;
            y[i].qs[j * 4 + 3] =
                std::arch::aarch64::vgetq_lane_s32(q_clamped, 3) as i8;
        }
    }
}

// Public interface
#[cfg(target_arch = "aarch64")]
pub fn quantize_q8_k(x: &[f32], y: &mut [BlockQ8_K]) {
    unsafe { quantize_row_q8_k_neon(x, y) }
}

#[cfg(target_arch = "aarch64")]
pub fn quantize_q8_0(x: &[f32], y: &mut [BlockQ8_0]) {
    unsafe { quantize_row_q8_0_neon(x, y) }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn quantize_q8_k(x: &[f32], y: &mut [BlockQ8_K]) {
    quantize_row_q8_k(x, y)
}

#[cfg(not(target_arch = "aarch64"))]
pub fn quantize_q8_0(x: &[f32], y: &mut [BlockQ8_0]) {
    quantize_row_q8_0(x, y)
}
