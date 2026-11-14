// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Matrix-vector multiplication (GEMV) for decode

use super::vec_dot::*;
use crate::types::*;

// Q4_0 GEMV
pub fn gemv_q4_0(
    weights: &[BlockQ4_0],
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let blocks_per_row = in_features / QK8_0;

    assert_eq!(weights.len(), out_features * blocks_per_row);
    assert_eq!(input.len(), blocks_per_row);
    assert_eq!(output.len(), out_features);

    for row in 0..out_features {
        let weight_row =
            &weights[row * blocks_per_row..(row + 1) * blocks_per_row];
        output[row] = vec_dot_q4_0_q8_0(weight_row, input);
    }
}

// Q8_0 GEMV
pub fn gemv_q8_0(
    weights: &[BlockQ8_0],
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let blocks_per_row = in_features / QK8_0;

    assert_eq!(weights.len(), out_features * blocks_per_row);
    assert_eq!(input.len(), blocks_per_row);
    assert_eq!(output.len(), out_features);

    for row in 0..out_features {
        let weight_row =
            &weights[row * blocks_per_row..(row + 1) * blocks_per_row];
        let mut sum = 0.0f32;

        for (w, x) in weight_row.iter().zip(input.iter()) {
            let dx = fp16_to_fp32(w.d);
            let dy = fp16_to_fp32(x.d);

            let mut dot = 0i32;
            for i in 0..32 {
                dot += (w.qs[i] as i32) * (x.qs[i] as i32);
            }

            sum += (dot as f32) * dx * dy;
        }

        output[row] = sum;
    }
}

// Q4_K GEMV
pub fn gemv_q4_k(
    weights: &[BlockQ4_K],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let blocks_per_row = in_features / QK_K;

    assert_eq!(weights.len(), out_features * blocks_per_row);
    assert_eq!(input.len(), blocks_per_row);
    assert_eq!(output.len(), out_features);

    for row in 0..out_features {
        let weight_row =
            &weights[row * blocks_per_row..(row + 1) * blocks_per_row];
        output[row] = vec_dot_q4_k_q8_k(weight_row, input);
    }
}

// Q5_K GEMV
pub fn gemv_q5_k(
    weights: &[BlockQ5_K],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let blocks_per_row = in_features / QK_K;

    assert_eq!(weights.len(), out_features * blocks_per_row);
    assert_eq!(input.len(), blocks_per_row);
    assert_eq!(output.len(), out_features);

    for row in 0..out_features {
        let weight_row =
            &weights[row * blocks_per_row..(row + 1) * blocks_per_row];
        output[row] = vec_dot_q5_k_q8_k(weight_row, input);
    }
}

// Q6_K GEMV
pub fn gemv_q6_k(
    weights: &[BlockQ6_K],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
) {
    let blocks_per_row = in_features / QK_K;

    assert_eq!(weights.len(), out_features * blocks_per_row);
    assert_eq!(input.len(), blocks_per_row);
    assert_eq!(output.len(), out_features);

    for row in 0..out_features {
        let weight_row =
            &weights[row * blocks_per_row..(row + 1) * blocks_per_row];
        output[row] = vec_dot_q6_k_q8_k(weight_row, input);
    }
}

// ============================================================================
// Transposed GEMV for GGUF weight layout
// GGUF stores weights as [in_features, out_features] (transposed)
// For decode: output[j] = Σ_i weight[i,j] * input[i]
//
// IMPORTANT: For quantized formats, the transposed layout is NOT compatible
// with standard vec_dot operations. GGUF uses a different storage scheme where
// weights are stored row-by-row for each output feature, but dims show the
// transposed shape. The actual memory layout is [out_features][in_features].
// ============================================================================

/// Helper function to compute vec_dot for a single block pair
/// This is used by transposed GEMV when input is quantized block-by-block
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q4_0_q8_0_single(x: &BlockQ4_0, y: &BlockQ8_0) -> f32 {
    unsafe { vec_dot_q4_0_q8_0_neon(&[*x], &[*y]) }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_dot_q4_0_q8_0_single(x: &BlockQ4_0, y: &BlockQ8_0) -> f32 {
    vec_dot_q4_0_q8_0(&[*x], &[*y])
}

/// Single block Q8_0 vec_dot
pub fn vec_dot_q8_0_single(x: &BlockQ8_0, y: &BlockQ8_0) -> f32 {
    let dx = fp16_to_fp32(x.d);
    let dy = fp16_to_fp32(y.d);
    let mut dot = 0i32;
    for i in 0..32 {
        dot += (x.qs[i] as i32) * (y.qs[i] as i32);
    }
    (dot as f32) * dx * dy
}

/// Single block Q4_K vec_dot
pub fn vec_dot_q4_k_q8_k_single(x: &BlockQ4_K, y: &BlockQ8_K) -> f32 {
    vec_dot_q4_k_q8_k(&[*x], &[*y])
}

/// Single block Q5_K vec_dot
pub fn vec_dot_q5_k_q8_k_single(x: &BlockQ5_K, y: &BlockQ8_K) -> f32 {
    vec_dot_q5_k_q8_k(&[*x], &[*y])
}

/// Single block Q6_K vec_dot
pub fn vec_dot_q6_k_q8_k_single(x: &BlockQ6_K, y: &BlockQ8_K) -> f32 {
    vec_dot_q6_k_q8_k(&[*x], &[*y])
}

// Note: Transposed GEMV for quantized weights requires different storage format.
// The current GGUF format stores weights as [out_features][in_features] in memory,
// but reports dims as [in_features, out_features]. Standard GEMV works correctly.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemv_q4_0() {
        let in_features = 64;
        let out_features = 32;
        let blocks_per_row = in_features / QK8_0;

        let mut weights = Vec::with_capacity(out_features * blocks_per_row);
        let mut input = Vec::with_capacity(blocks_per_row);

        for _i in 0..out_features * blocks_per_row {
            weights.push(BlockQ4_0 { d: fp32_to_fp16(0.1), qs: [0x88; 16] });
        }

        for _i in 0..blocks_per_row {
            input.push(BlockQ8_0 { d: fp32_to_fp16(1.0), qs: [1; 32] });
        }

        let mut output = vec![0.0f32; out_features];

        gemv_q4_0(&weights, &input, &mut output, out_features, in_features);

        for &v in &output {
            assert!(v.abs() < 1.0);
        }
    }
}
