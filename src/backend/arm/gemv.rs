// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// GEMV with block-tiling for cache locality optimization
// Based on llama.cpp's ggml-cpu.c block-tiling approach

use crate::types::{
    BlockIQ4XS, BlockQ4_0, BlockQ4_K, BlockQ5_K, BlockQ6_K, BlockQ8_0,
    BlockQ8_K, QK_K,
};

use super::vec_dot::{
    vec_dot_iq4_xs_q8_k, vec_dot_q4_0_q8_0, vec_dot_q4_k_q8_k,
    vec_dot_q5_k_q8_k, vec_dot_q6_k_q8_k,
};

pub const QK4_0: usize = 32;

// Block-tiling size from llama.cpp
const BLOCK_ROWS: usize = 16;

// ============================================================================
// Generic block-tiling GEMV implementation
// ============================================================================

/// Block-tiling GEMV for K-quant formats
/// Processes weights in 16-row blocks to maximize cache locality
#[inline(always)]
unsafe fn gemv_block_tiling<W, F>(
    weight_data: &[u8],
    row_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    nb: usize,
    vec_dot: F,
    n_threads: usize,
) where
    W: Copy,
    F: Fn(&[W], &[BlockQ8_K]) -> f32 + Sync + Send + Copy,
{
    if n_threads > 1 && out_features >= BLOCK_ROWS * 2 {
        // Parallel processing using thread::scope for low overhead
        // Use usize to store pointers (Send) and convert back in closure
        let output_ptr = output.as_mut_ptr() as usize;
        let weight_ptr = weight_data.as_ptr() as usize;
        let input_ptr = input.as_ptr() as usize;
        let input_len = input.len();

        std::thread::scope(|s| {
            let chunk_size = (out_features + n_threads - 1) / n_threads;
            for tid in 0..n_threads {
                let start = tid * chunk_size;
                let end = (start + chunk_size).min(out_features);
                if start >= out_features {
                    break;
                }

                s.spawn(move || {
                    let output_ptr = output_ptr as *mut f32;
                    let weight_ptr = weight_ptr as *const u8;
                    let input_ptr = input_ptr as *const BlockQ8_K;

                    for i in start..end {
                        let row_blocks: &[W] = unsafe {
                            std::slice::from_raw_parts(
                                weight_ptr.add(i * row_stride) as *const W,
                                nb,
                            )
                        };
                        let input_slice = unsafe {
                            std::slice::from_raw_parts(input_ptr, input_len)
                        };
                        unsafe {
                            *output_ptr.add(i) =
                                vec_dot(row_blocks, input_slice);
                        }
                    }
                });
            }
        });
    } else {
        // Sequential processing with block-tiling
        for block_start in (0..out_features).step_by(BLOCK_ROWS) {
            let block_end = (block_start + BLOCK_ROWS).min(out_features);

            // Process a block of rows together to maximize input reuse
            for i in block_start..block_end {
                let row_blocks: &[W] = std::slice::from_raw_parts(
                    weight_data.as_ptr().add(i * row_stride) as *const W,
                    nb,
                );
                output[i] = vec_dot(row_blocks, input);
            }
        }
    }
}

// ============================================================================
// Single-token GEMV (decode path)
// ============================================================================

/// GEMV for Q4_K with block-tiling
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

    gemv_block_tiling::<BlockQ4_K, _>(
        weight_data,
        row_stride,
        input,
        output,
        out_features,
        nb,
        vec_dot_q4_k_q8_k,
        n_threads,
    );
}

/// GEMV for Q5_K with block-tiling
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

    gemv_block_tiling::<BlockQ5_K, _>(
        weight_data,
        row_stride,
        input,
        output,
        out_features,
        nb,
        vec_dot_q5_k_q8_k,
        n_threads,
    );
}

/// GEMV for Q6_K with block-tiling
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

    gemv_block_tiling::<BlockQ6_K, _>(
        weight_data,
        row_stride,
        input,
        output,
        out_features,
        nb,
        vec_dot_q6_k_q8_k,
        n_threads,
    );
}

/// GEMV for IQ4_XS with block-tiling
pub unsafe fn gemv_iq4_xs_transposed(
    weight_data: &[u8],
    _weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_threads: usize,
) {
    let nb = in_features / QK_K;
    let row_stride = nb * std::mem::size_of::<BlockIQ4XS>();

    gemv_block_tiling::<BlockIQ4XS, _>(
        weight_data,
        row_stride,
        input,
        output,
        out_features,
        nb,
        vec_dot_iq4_xs_q8_k,
        n_threads,
    );
}

/// GEMV for Q4_0
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

    for block_start in (0..out_features).step_by(BLOCK_ROWS) {
        let block_end = (block_start + BLOCK_ROWS).min(out_features);
        for i in block_start..block_end {
            let col_offset = i * weight_stride;
            let col_blocks: &[BlockQ4_0] = std::slice::from_raw_parts(
                weight_data.as_ptr().add(col_offset) as *const BlockQ4_0,
                nb,
            );
            output[i] = vec_dot_q4_0_q8_0(col_blocks, input);
        }
    }
}

// ============================================================================
// Batch GEMV (prefill path)
// ============================================================================

/// Batched GEMV for Q4_K with block-tiling
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
        // Parallel processing using thread::scope
        let total_outputs = n_tokens * out_features;
        let output_ptr = output.as_mut_ptr() as usize;
        let weight_ptr = weight_data.as_ptr() as usize;
        let input_ptr = input.as_ptr() as usize;

        std::thread::scope(|s| {
            let chunk_size = (total_outputs + n_threads - 1) / n_threads;
            for tid in 0..n_threads {
                let start = tid * chunk_size;
                let end = (start + chunk_size).min(total_outputs);
                if start >= total_outputs {
                    break;
                }

                s.spawn(move || {
                    let output_ptr = output_ptr as *mut f32;
                    let weight_ptr = weight_ptr as *const u8;
                    let input_ptr = input_ptr as *const BlockQ8_K;

                    for idx in start..end {
                        let t = idx / out_features;
                        let i = idx % out_features;
                        let input_slice = unsafe {
                            std::slice::from_raw_parts(
                                input_ptr.add(t * nb),
                                nb,
                            )
                        };
                        let row_blocks: &[BlockQ4_K] = unsafe {
                            std::slice::from_raw_parts(
                                weight_ptr.add(i * row_stride)
                                    as *const BlockQ4_K,
                                nb,
                            )
                        };
                        unsafe {
                            *output_ptr.add(idx) =
                                vec_dot_q4_k_q8_k(row_blocks, input_slice);
                        }
                    }
                });
            }
        });
    } else {
        for t in 0..n_tokens {
            let input_slice = &input[t * nb..(t + 1) * nb];
            for block_start in (0..out_features).step_by(BLOCK_ROWS) {
                let block_end = (block_start + BLOCK_ROWS).min(out_features);
                for i in block_start..block_end {
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
}

/// Batched GEMV for Q5_K with block-tiling
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
        let total_outputs = n_tokens * out_features;
        let output_ptr = output.as_mut_ptr() as usize;
        let weight_ptr = weight_data.as_ptr() as usize;
        let input_ptr = input.as_ptr() as usize;

        std::thread::scope(|s| {
            let chunk_size = (total_outputs + n_threads - 1) / n_threads;
            for tid in 0..n_threads {
                let start = tid * chunk_size;
                let end = (start + chunk_size).min(total_outputs);
                if start >= total_outputs {
                    break;
                }

                s.spawn(move || {
                    let output_ptr = output_ptr as *mut f32;
                    let weight_ptr = weight_ptr as *const u8;
                    let input_ptr = input_ptr as *const BlockQ8_K;

                    for idx in start..end {
                        let t = idx / out_features;
                        let i = idx % out_features;
                        let input_slice = unsafe {
                            std::slice::from_raw_parts(
                                input_ptr.add(t * nb),
                                nb,
                            )
                        };
                        let row_blocks: &[BlockQ5_K] = unsafe {
                            std::slice::from_raw_parts(
                                weight_ptr.add(i * row_stride)
                                    as *const BlockQ5_K,
                                nb,
                            )
                        };
                        unsafe {
                            *output_ptr.add(idx) =
                                vec_dot_q5_k_q8_k(row_blocks, input_slice);
                        }
                    }
                });
            }
        });
    } else {
        for t in 0..n_tokens {
            let input_slice = &input[t * nb..(t + 1) * nb];
            for block_start in (0..out_features).step_by(BLOCK_ROWS) {
                let block_end = (block_start + BLOCK_ROWS).min(out_features);
                for i in block_start..block_end {
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
}

/// Batched GEMV for Q6_K with block-tiling
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
        let total_outputs = n_tokens * out_features;
        let output_ptr = output.as_mut_ptr() as usize;
        let weight_ptr = weight_data.as_ptr() as usize;
        let input_ptr = input.as_ptr() as usize;

        std::thread::scope(|s| {
            let chunk_size = (total_outputs + n_threads - 1) / n_threads;
            for tid in 0..n_threads {
                let start = tid * chunk_size;
                let end = (start + chunk_size).min(total_outputs);
                if start >= total_outputs {
                    break;
                }

                s.spawn(move || {
                    let output_ptr = output_ptr as *mut f32;
                    let weight_ptr = weight_ptr as *const u8;
                    let input_ptr = input_ptr as *const BlockQ8_K;

                    for idx in start..end {
                        let t = idx / out_features;
                        let i = idx % out_features;
                        let input_slice = unsafe {
                            std::slice::from_raw_parts(
                                input_ptr.add(t * nb),
                                nb,
                            )
                        };
                        let row_blocks: &[BlockQ6_K] = unsafe {
                            std::slice::from_raw_parts(
                                weight_ptr.add(i * row_stride)
                                    as *const BlockQ6_K,
                                nb,
                            )
                        };
                        unsafe {
                            *output_ptr.add(idx) =
                                vec_dot_q6_k_q8_k(row_blocks, input_slice);
                        }
                    }
                });
            }
        });
    } else {
        for t in 0..n_tokens {
            let input_slice = &input[t * nb..(t + 1) * nb];
            for block_start in (0..out_features).step_by(BLOCK_ROWS) {
                let block_end = (block_start + BLOCK_ROWS).min(out_features);
                for i in block_start..block_end {
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
}

/// Batched GEMV for IQ4_XS with block-tiling
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
        let total_outputs = n_tokens * out_features;
        let output_ptr = output.as_mut_ptr() as usize;
        let weight_ptr = weight_data.as_ptr() as usize;
        let input_ptr = input.as_ptr() as usize;

        std::thread::scope(|s| {
            let chunk_size = (total_outputs + n_threads - 1) / n_threads;
            for tid in 0..n_threads {
                let start = tid * chunk_size;
                let end = (start + chunk_size).min(total_outputs);
                if start >= total_outputs {
                    break;
                }

                s.spawn(move || {
                    let output_ptr = output_ptr as *mut f32;
                    let weight_ptr = weight_ptr as *const u8;
                    let input_ptr = input_ptr as *const BlockQ8_K;

                    for idx in start..end {
                        let t = idx / out_features;
                        let i = idx % out_features;
                        let input_slice = unsafe {
                            std::slice::from_raw_parts(
                                input_ptr.add(t * nb),
                                nb,
                            )
                        };
                        let row_blocks: &[BlockIQ4XS] = unsafe {
                            std::slice::from_raw_parts(
                                weight_ptr.add(i * row_stride)
                                    as *const BlockIQ4XS,
                                nb,
                            )
                        };
                        unsafe {
                            *output_ptr.add(idx) =
                                vec_dot_iq4_xs_q8_k(row_blocks, input_slice);
                        }
                    }
                });
            }
        });
    } else {
        for t in 0..n_tokens {
            let input_slice = &input[t * nb..(t + 1) * nb];
            for block_start in (0..out_features).step_by(BLOCK_ROWS) {
                let block_end = (block_start + BLOCK_ROWS).min(out_features);
                for i in block_start..block_end {
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
}

/// Batched GEMV for Q4_0
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
        for block_start in (0..out_features).step_by(BLOCK_ROWS) {
            let block_end = (block_start + BLOCK_ROWS).min(out_features);
            for i in block_start..block_end {
                let col_offset = i * weight_stride;
                let col_blocks: &[BlockQ4_0] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr().add(col_offset)
                            as *const BlockQ4_0,
                        nb,
                    )
                };
                output[t * out_features + i] =
                    vec_dot_q4_0_q8_0(col_blocks, input_slice);
            }
        }
    }
}
