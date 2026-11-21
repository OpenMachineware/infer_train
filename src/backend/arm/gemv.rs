// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Matrix-vector multiplication (GEMV) for decode
// Uses tiled implementation from llama.cpp

use super::tiled::{
    mul_mat_q4_k_tiled, mul_mat_q5_k_tiled, mul_mat_q6_k_tiled,
};
use super::vec_dot::*;
use crate::types::*;

// ============================================================================
// Q4_0, Q8_0, Q4_K: Keep simple vec_dot implementation (no tiled version yet)
// ============================================================================

/// Q4_0 GEMV
pub fn gemv_q4_0(
    weights: &[BlockQ4_0],
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
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

/// Q8_0 GEMV
pub fn gemv_q8_0(
    weights: &[BlockQ8_0],
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
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

/// Q4_K GEMV (simple vec_dot, no tiled version)
pub fn gemv_q4_k(
    weights: &[BlockQ4_K],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
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

/// Q5_K GEMV (simple vec_dot for n_tokens=1)
pub fn gemv_q5_k(
    weights: &[BlockQ5_K],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
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

/// Q6_K GEMV (simple vec_dot for n_tokens=1)
pub fn gemv_q6_k(
    weights: &[BlockQ6_K],
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
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
// Transposed GEMV for GGUF weight layout (uses tiled implementation)
// ============================================================================

/// Matrix multiplication for Q6_K weights
/// Dispatch based on batch size: vec_dot for decode (n<8), tiled for prefill (n>=8)
pub fn mul_mat_q6_k(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) {
    if n_tokens < 8 {
        // Decode path: use vec_dot (memory-bound optimization)
        mul_mat_q6_k_vec_dot(
            weights,
            col_stride,
            input,
            output,
            out_features,
            in_features,
            n_tokens,
            n_threads,
        );
    } else {
        // Prefill path: use tiled (compute-bound optimization)
        mul_mat_q6_k_tiled(
            weights,
            col_stride,
            input,
            output,
            out_features,
            in_features,
            n_tokens,
        );
    }
}

/// Matrix multiplication for Q5_K weights
/// Dispatch based on batch size: vec_dot for decode (n<8), tiled for prefill (n>=8)
pub fn mul_mat_q5_k(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) {
    if n_tokens < 8 {
        // Decode path: use vec_dot (memory-bound optimization)
        mul_mat_q5_k_vec_dot(
            weights,
            col_stride,
            input,
            output,
            out_features,
            in_features,
            n_tokens,
            n_threads,
        );
    } else {
        // Prefill path: use tiled (compute-bound optimization)
        mul_mat_q5_k_tiled(
            weights,
            col_stride,
            input,
            output,
            out_features,
            in_features,
            n_tokens,
        );
    }
}

/// Q4_K transposed GEMV
/// Dispatch based on batch size: vec_dot for decode (n<8), tiled for prefill (n>=8)
/// Note: Tiled path disabled for now due to scales encoding complexity
pub fn mul_mat_q4_k(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) {
    // Always use vec_dot for now - tiled needs correct scales handling
    mul_mat_q4_k_vec_dot(
        weights,
        col_stride,
        input,
        output,
        out_features,
        in_features,
        n_tokens,
    );
}

/// Q4_K vec_dot path for decode
pub fn mul_mat_q4_k_vec_dot(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
) {
    let n_blocks = in_features / QK_K;

    assert_eq!(input.len(), n_tokens * n_blocks);
    assert_eq!(output.len(), n_tokens * out_features);

    output.fill(0.0);

    for token_idx in 0..n_tokens {
        let input_start = token_idx * n_blocks;
        let input_row = &input[input_start..input_start + n_blocks];

        for row in 0..out_features {
            let row_offset = row * col_stride;
            let weight_row = unsafe {
                std::slice::from_raw_parts(
                    weights.as_ptr().add(row_offset) as *const BlockQ4_K,
                    n_blocks,
                )
            };
            output[token_idx * out_features + row] =
                vec_dot_q4_k_q8_k(weight_row, input_row);
        }
    }
}

/// IQ4_XS transposed GEMV (no tiled version, use vec_dot)
pub fn mul_mat_iq4_xs(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    _n_threads: usize,
) {
    use super::vec_dot::vec_dot_iq4_xs_q8_k;
    use crate::types::BlockIQ4XS;

    let n_blocks = in_features / QK_K;

    assert_eq!(input.len(), n_tokens * n_blocks);
    assert_eq!(output.len(), n_tokens * out_features);

    output.fill(0.0);

    for token_idx in 0..n_tokens {
        let input_start = token_idx * n_blocks;
        let input_row = &input[input_start..input_start + n_blocks];

        for row in 0..out_features {
            let row_offset = row * col_stride;
            let weight_row = unsafe {
                std::slice::from_raw_parts(
                    weights.as_ptr().add(row_offset) as *const BlockIQ4XS,
                    n_blocks,
                )
            };
            output[token_idx * out_features + row] =
                vec_dot_iq4_xs_q8_k(weight_row, input_row);
        }
    }
}

/// Q5_K transposed GEMV using vec_dot (decode path, memory-bound)
/// Multi-threaded for large matrices, single-thread for small
pub fn mul_mat_q5_k_vec_dot(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) {
    let n_blocks = in_features / QK_K;

    assert_eq!(input.len(), n_tokens * n_blocks);
    assert_eq!(output.len(), n_tokens * out_features);

    output.fill(0.0);

    // For decode (n_tokens=1), limit threads for cache efficiency
    // For prefill (n_tokens>1), use more threads
    let effective_threads = if n_tokens == 1 {
        // decode: small m, limit threads
        if out_features < 1024 {
            1
        } else if out_features < 2048 {
            2
        } else {
            n_threads.min(4)
        }
    } else {
        // prefill: use all threads
        n_threads
    };

    if effective_threads <= 1 {
        // Single-thread path
        for token_idx in 0..n_tokens {
            let input_start = token_idx * n_blocks;
            let input_row = &input[input_start..input_start + n_blocks];

            for row in 0..out_features {
                let row_offset = row * col_stride;
                let weight_row = unsafe {
                    std::slice::from_raw_parts(
                        weights.as_ptr().add(row_offset) as *const BlockQ5_K,
                        n_blocks,
                    )
                };
                output[token_idx * out_features + row] =
                    vec_dot_q5_k_q8_k(weight_row, input_row);
            }
        }
    } else {
        // Multi-thread path: parallelize over output rows
        use std::thread;
        let output_ptr = output.as_mut_ptr() as usize;
        let chunk_size =
            (out_features + effective_threads - 1) / effective_threads;
        thread::scope(|s| {
            for tid in 0..effective_threads {
                let start = tid * chunk_size;
                let end = (start + chunk_size).min(out_features);
                if start >= out_features {
                    break;
                }
                s.spawn(move || {
                    let output_ptr = output_ptr as *mut f32;
                    for row in start..end {
                        let row_offset = row * col_stride;
                        let weight_row = unsafe {
                            std::slice::from_raw_parts(
                                weights.as_ptr().add(row_offset)
                                    as *const BlockQ5_K,
                                n_blocks,
                            )
                        };
                        for token_idx in 0..n_tokens {
                            let input_start = token_idx * n_blocks;
                            let input_row =
                                &input[input_start..input_start + n_blocks];
                            unsafe {
                                *output_ptr
                                    .add(token_idx * out_features + row) =
                                    vec_dot_q5_k_q8_k(weight_row, input_row);
                            }
                        }
                    }
                });
            }
        });
    }
}

/// Q6_K transposed GEMV using vec_dot (decode path, memory-bound)
/// Multi-threaded for large matrices, single-thread for small
pub fn mul_mat_q6_k_vec_dot(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) {
    let n_blocks = in_features / QK_K;

    assert_eq!(input.len(), n_tokens * n_blocks);
    assert_eq!(output.len(), n_tokens * out_features);

    output.fill(0.0);

    let effective_threads = if n_tokens == 1 {
        if out_features < 1024 {
            1
        } else if out_features < 2048 {
            2
        } else {
            n_threads.min(4)
        }
    } else {
        n_threads
    };

    if effective_threads <= 1 {
        for token_idx in 0..n_tokens {
            let input_start = token_idx * n_blocks;
            let input_row = &input[input_start..input_start + n_blocks];

            for row in 0..out_features {
                let row_offset = row * col_stride;
                let weight_row = unsafe {
                    std::slice::from_raw_parts(
                        weights.as_ptr().add(row_offset) as *const BlockQ6_K,
                        n_blocks,
                    )
                };
                output[token_idx * out_features + row] =
                    vec_dot_q6_k_q8_k(weight_row, input_row);
            }
        }
    } else {
        // Multi-thread path: parallelize over output rows
        use std::thread;
        let output_ptr = output.as_mut_ptr() as usize;
        let chunk_size =
            (out_features + effective_threads - 1) / effective_threads;
        thread::scope(|s| {
            for tid in 0..effective_threads {
                let start = tid * chunk_size;
                let end = (start + chunk_size).min(out_features);
                if start >= out_features {
                    break;
                }
                s.spawn(move || {
                    let output_ptr = output_ptr as *mut f32;
                    for row in start..end {
                        let row_offset = row * col_stride;
                        let weight_row = unsafe {
                            std::slice::from_raw_parts(
                                weights.as_ptr().add(row_offset)
                                    as *const BlockQ6_K,
                                n_blocks,
                            )
                        };
                        for token_idx in 0..n_tokens {
                            let input_start = token_idx * n_blocks;
                            let input_row =
                                &input[input_start..input_start + n_blocks];
                            unsafe {
                                *output_ptr
                                    .add(token_idx * out_features + row) =
                                    vec_dot_q6_k_q8_k(weight_row, input_row);
                            }
                        }
                    }
                });
            }
        });
    }
}

/// Q4_0 transposed GEMV
pub fn mul_mat_q4_0(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    _n_threads: usize,
) {
    let n_blocks = in_features / QK8_0;

    assert_eq!(input.len(), n_tokens * n_blocks);
    assert_eq!(output.len(), n_tokens * out_features);

    output.fill(0.0);

    for token_idx in 0..n_tokens {
        let input_start = token_idx * n_blocks;
        let input_row = &input[input_start..input_start + n_blocks];

        for row in 0..out_features {
            let row_offset = row * col_stride;
            let weight_row = unsafe {
                std::slice::from_raw_parts(
                    weights.as_ptr().add(row_offset) as *const BlockQ4_0,
                    n_blocks,
                )
            };
            output[token_idx * out_features + row] =
                vec_dot_q4_0_q8_0(weight_row, input_row);
        }
    }
}

// Legacy aliases
pub fn gemv_q4_k_transposed(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_threads: usize,
) {
    mul_mat_q4_k(
        weights,
        col_stride,
        input,
        output,
        out_features,
        in_features,
        1,
        n_threads,
    )
}

pub fn gemv_q5_k_transposed(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_threads: usize,
) {
    mul_mat_q5_k(
        weights,
        col_stride,
        input,
        output,
        out_features,
        in_features,
        1,
        n_threads,
    )
}

pub fn gemv_q6_k_transposed(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_threads: usize,
) {
    mul_mat_q6_k(
        weights,
        col_stride,
        input,
        output,
        out_features,
        in_features,
        1,
        n_threads,
    )
}

pub fn gemv_q4_0_transposed(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_threads: usize,
) {
    mul_mat_q4_0(
        weights,
        col_stride,
        input,
        output,
        out_features,
        in_features,
        1,
        n_threads,
    )
}
