// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// GEMV operations for quantized matrix-vector multiplication

use crate::types::{
    BlockIQ4XS, BlockQ4_0, BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_0,
    BlockQ8_K, QK_K,
};

use super::vec_dot::{
    vec_dot_iq4_xs_q8_k, vec_dot_q4_0_q8_0, vec_dot_q4_k_q8_k,
    vec_dot_q5_k_q8_k, vec_dot_q6_k_q8_k,
};

pub const QK4_0: usize = 32;

// ============================================================================
// Single-token GEMV (decode path)
// ============================================================================

/// GEMV for Q4_K with transposed weights (row-major storage)
pub unsafe fn gemv_q4_k_transposed(
    weight_data: &[u8],
    _weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_threads: usize,
) {
    let nb = in_features / QK_K;
    let row_stride = nb * std::mem::size_of::<BlockQ4_K>();

    if n_threads > 1 && out_features >= 64 {
        use rayon::prelude::*;
        let results: Vec<f32> = (0..out_features)
            .into_par_iter()
            .map(|i| {
                let row_blocks: &[BlockQ4_K] = std::slice::from_raw_parts(
                    weight_data.as_ptr().add(i * row_stride)
                        as *const BlockQ4_K,
                    nb,
                );
                vec_dot_q4_k_q8_k(row_blocks, input)
            })
            .collect();
        output.copy_from_slice(&results);
    } else {
        for i in 0..out_features {
            let row_blocks: &[BlockQ4_K] = std::slice::from_raw_parts(
                weight_data.as_ptr().add(i * row_stride) as *const BlockQ4_K,
                nb,
            );
            output[i] = vec_dot_q4_k_q8_k(row_blocks, input);
        }
    }
}

/// GEMV for Q5_K with transposed weights (row-major storage)
pub unsafe fn gemv_q5_k_transposed(
    weight_data: &[u8],
    _weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_threads: usize,
) {
    let nb = in_features / QK_K;
    let row_stride = nb * std::mem::size_of::<BlockQ5_K>();

    if n_threads > 1 && out_features >= 64 {
        use rayon::prelude::*;
        let results: Vec<f32> = (0..out_features)
            .into_par_iter()
            .map(|i| {
                let row_blocks: &[BlockQ5_K] = std::slice::from_raw_parts(
                    weight_data.as_ptr().add(i * row_stride)
                        as *const BlockQ5_K,
                    nb,
                );
                vec_dot_q5_k_q8_k(row_blocks, input)
            })
            .collect();
        output.copy_from_slice(&results);
    } else {
        for i in 0..out_features {
            let row_blocks: &[BlockQ5_K] = std::slice::from_raw_parts(
                weight_data.as_ptr().add(i * row_stride) as *const BlockQ5_K,
                nb,
            );
            output[i] = vec_dot_q5_k_q8_k(row_blocks, input);
        }
    }
}

/// GEMV for Q6_K with transposed weights (row-major storage)
pub unsafe fn gemv_q6_k_transposed(
    weight_data: &[u8],
    _weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_threads: usize,
) {
    let nb = in_features / QK_K;
    let row_stride = nb * std::mem::size_of::<BlockQ6_K>();

    if n_threads > 1 && out_features >= 64 {
        use rayon::prelude::*;
        let results: Vec<f32> = (0..out_features)
            .into_par_iter()
            .map(|i| {
                let row_blocks: &[BlockQ6_K] = std::slice::from_raw_parts(
                    weight_data.as_ptr().add(i * row_stride)
                        as *const BlockQ6_K,
                    nb,
                );
                vec_dot_q6_k_q8_k(row_blocks, input)
            })
            .collect();
        output.copy_from_slice(&results);
    } else {
        for i in 0..out_features {
            let row_blocks: &[BlockQ6_K] = std::slice::from_raw_parts(
                weight_data.as_ptr().add(i * row_stride) as *const BlockQ6_K,
                nb,
            );
            output[i] = vec_dot_q6_k_q8_k(row_blocks, input);
        }
    }
}

/// GEMV for Q4_0 with transposed weights
pub unsafe fn gemv_q4_0_transposed(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    _n_threads: usize,
) {
    let nb = in_features / QK4_0;

    for i in 0..out_features {
        let col_offset = i * weight_stride;
        let col_blocks: &[BlockQ4_0] = std::slice::from_raw_parts(
            weight_data.as_ptr().add(col_offset) as *const BlockQ4_0,
            nb,
        );
        output[i] = vec_dot_q4_0_q8_0(col_blocks, input);
    }
}

// ============================================================================
// Batch GEMV (prefill path)
// ============================================================================

/// Batched matrix multiplication for Q4_0
pub fn mul_mat_q4_0(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    _n_threads: usize,
) {
    let nb = in_features / QK4_0;

    for t in 0..n_tokens {
        let input_slice = &input[t * nb..(t + 1) * nb];
        for i in 0..out_features {
            let col_offset = i * weight_stride;
            let col_blocks: &[BlockQ4_0] = unsafe {
                std::slice::from_raw_parts(
                    weight_data.as_ptr().add(col_offset) as *const BlockQ4_0,
                    nb,
                )
            };
            output[t * out_features + i] =
                vec_dot_q4_0_q8_0(col_blocks, input_slice);
        }
    }
}

/// Batched matrix multiplication for Q5_K
pub fn mul_mat_q5_k(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) {
    let nb = in_features / QK_K;
    let row_stride = if weight_stride > 0 {
        weight_stride
    } else {
        nb * std::mem::size_of::<BlockQ5_K>()
    };

    if n_threads > 1 && (n_tokens * out_features) >= 64 {
        use rayon::prelude::*;
        let results: Vec<f32> = (0..n_tokens * out_features)
            .into_par_iter()
            .map(|idx| {
                let t = idx / out_features;
                let i = idx % out_features;
                let input_slice = &input[t * nb..(t + 1) * nb];
                let row_blocks: &[BlockQ5_K] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr().add(i * row_stride)
                            as *const BlockQ5_K,
                        nb,
                    )
                };
                vec_dot_q5_k_q8_k(row_blocks, input_slice)
            })
            .collect();
        output.copy_from_slice(&results);
    } else {
        for t in 0..n_tokens {
            let input_slice = &input[t * nb..(t + 1) * nb];
            for i in 0..out_features {
                let row_blocks: &[BlockQ5_K] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr().add(i * row_stride)
                            as *const BlockQ5_K,
                        nb,
                    )
                };
                output[t * out_features + i] =
                    vec_dot_q5_k_q8_k(row_blocks, input_slice);
            }
        }
    }
}

/// Batched matrix multiplication for Q4_K
pub fn mul_mat_q4_k(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) {
    let nb = in_features / QK_K;
    let row_stride = if weight_stride > 0 {
        weight_stride
    } else {
        nb * std::mem::size_of::<BlockQ4_K>()
    };

    if n_threads > 1 && (n_tokens * out_features) >= 64 {
        use rayon::prelude::*;
        let results: Vec<f32> = (0..n_tokens * out_features)
            .into_par_iter()
            .map(|idx| {
                let t = idx / out_features;
                let i = idx % out_features;
                let input_slice = &input[t * nb..(t + 1) * nb];
                let row_blocks: &[BlockQ4_K] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr().add(i * row_stride)
                            as *const BlockQ4_K,
                        nb,
                    )
                };
                vec_dot_q4_k_q8_k(row_blocks, input_slice)
            })
            .collect();
        output.copy_from_slice(&results);
    } else {
        for t in 0..n_tokens {
            let input_slice = &input[t * nb..(t + 1) * nb];
            for i in 0..out_features {
                let row_blocks: &[BlockQ4_K] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr().add(i * row_stride)
                            as *const BlockQ4_K,
                        nb,
                    )
                };
                output[t * out_features + i] =
                    vec_dot_q4_k_q8_k(row_blocks, input_slice);
            }
        }
    }
}

/// Batched matrix multiplication for Q6_K
pub fn mul_mat_q6_k(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) {
    let nb = in_features / QK_K;
    let row_stride = if weight_stride > 0 {
        weight_stride
    } else {
        nb * std::mem::size_of::<BlockQ6_K>()
    };

    if n_threads > 1 && (n_tokens * out_features) >= 64 {
        use rayon::prelude::*;
        let results: Vec<f32> = (0..n_tokens * out_features)
            .into_par_iter()
            .map(|idx| {
                let t = idx / out_features;
                let i = idx % out_features;
                let input_slice = &input[t * nb..(t + 1) * nb];
                let row_blocks: &[BlockQ6_K] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr().add(i * row_stride)
                            as *const BlockQ6_K,
                        nb,
                    )
                };
                vec_dot_q6_k_q8_k(row_blocks, input_slice)
            })
            .collect();
        output.copy_from_slice(&results);
    } else {
        for t in 0..n_tokens {
            let input_slice = &input[t * nb..(t + 1) * nb];
            for i in 0..out_features {
                let row_blocks: &[BlockQ6_K] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr().add(i * row_stride)
                            as *const BlockQ6_K,
                        nb,
                    )
                };
                output[t * out_features + i] =
                    vec_dot_q6_k_q8_k(row_blocks, input_slice);
            }
        }
    }
}

/// Batched matrix multiplication for IQ4_XS
pub fn mul_mat_iq4_xs(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) {
    let nb = in_features / QK_K;
    let row_stride = if weight_stride > 0 {
        weight_stride
    } else {
        nb * std::mem::size_of::<BlockIQ4XS>()
    };

    if n_threads > 1 && (n_tokens * out_features) >= 64 {
        use rayon::prelude::*;
        let results: Vec<f32> = (0..n_tokens * out_features)
            .into_par_iter()
            .map(|idx| {
                let t = idx / out_features;
                let i = idx % out_features;
                let input_slice = &input[t * nb..(t + 1) * nb];
                let row_blocks: &[BlockIQ4XS] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr().add(i * row_stride)
                            as *const BlockIQ4XS,
                        nb,
                    )
                };
                vec_dot_iq4_xs_q8_k(row_blocks, input_slice)
            })
            .collect();
        output.copy_from_slice(&results);
    } else {
        for t in 0..n_tokens {
            let input_slice = &input[t * nb..(t + 1) * nb];
            for i in 0..out_features {
                let row_blocks: &[BlockIQ4XS] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr().add(i * row_stride)
                            as *const BlockIQ4XS,
                        nb,
                    )
                };
                output[t * out_features + i] =
                    vec_dot_iq4_xs_q8_k(row_blocks, input_slice);
            }
        }
    }
}
