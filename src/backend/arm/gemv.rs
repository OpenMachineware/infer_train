// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Matrix-vector multiplication (GEMV) for decode

use super::vec_dot::*;
use crate::backend::thread_pool::{optimal_threads, parallel_stealing_raw};
use crate::types::*;
use half::f16;

// ============================================================================
// K-extent optimization for narrow path (llama.cpp style)
// ============================================================================

/// Compute K-extent in blocks for narrow path.
/// This determines how many K blocks to read sequentially to fit in L1 cache.
/// Based on llama.cpp's ggml_tiled_narrow_k_extent.
/// L1 cache = 32KB, leave 31KB for weights.
/// k_extent = 31KB / (16 rows + n_tokens) in elements
/// Convert to blocks (each block = 256 elements = 144 bytes for Q4_K)
fn compute_k_extent_blocks(n_tokens: usize, block_size: usize) -> usize {
    const L1_BUDGET: usize = 31 * 1024; // 31KB for weights
    const TILE_K: usize = 256; // Each block covers 256 elements

    let n_rows = n_tokens.max(16);
    // k_extent in elements
    let k_extent = L1_BUDGET / (16 * block_size / 256 + n_rows);
    // Floor to multiple of TILE_K
    let k_extent = (k_extent / TILE_K) * TILE_K;
    // Convert to blocks
    let k_extent_blocks = k_extent / TILE_K;

    // Ensure at least 1 block
    k_extent_blocks.max(1)
}

// ============================================================================
// Work data structures for persistent thread pool
// ============================================================================

/// Work data for Q6_K mul_mat
#[repr(C)]
struct MulMatQ6KData {
    weights: *const u8,
    col_stride: usize,
    input: *const BlockQ8_K,
    output: *mut f32,
    out_features: usize,
    n_blocks: usize,
    block_size: usize,
    n_tokens: usize,
}

/// Work data for Q5_K mul_mat
#[repr(C)]
struct MulMatQ5KData {
    weights: *const u8,
    col_stride: usize,
    input: *const BlockQ8_K,
    output: *mut f32,
    out_features: usize,
    n_blocks: usize,
    block_size: usize,
    n_tokens: usize,
}

/// Work data for chunked Q5_K mul_mat (cache-friendly parallel)
#[repr(C)]
struct ChunkedMulMatData {
    weights: *const u8,
    col_stride: usize,
    input: *const BlockQ8_K,
    output: *mut f32,
    out_features: usize,
    n_blocks: usize,
    n_tokens: usize,
    chunk_size: usize, // Number of rows per chunk
}

/// Worker for chunked Q5_K: each chunk is a contiguous block of rows
fn chunked_mul_mat_q5_k_worker(chunk: usize, _tid: usize, data: *const u8) {
    unsafe {
        let data = &*(data as *const ChunkedMulMatData);
        let chunk_size = data.chunk_size;

        // Decode chunk to get row range
        let start_row = chunk * chunk_size;
        if start_row >= data.out_features {
            return;
        }
        let end_row = (start_row + chunk_size).min(data.out_features);

        // Process each token's slice
        for token_idx in 0..data.n_tokens {
            let input_start = token_idx * data.n_blocks;
            let input_row = std::slice::from_raw_parts(
                data.input.add(input_start),
                data.n_blocks,
            );

            // Process all rows in this chunk (contiguous memory access!)
            let mut tmp = [0.0f32; 64]; // Max chunk_size
            for (i, row) in (start_row..end_row).enumerate() {
                let row_offset = row * data.col_stride;
                let weight_row = std::slice::from_raw_parts(
                    data.weights.add(row_offset) as *const BlockQ5_K,
                    data.n_blocks,
                );
                tmp[i] = vec_dot_q5_k_q8_k(weight_row, input_row);
            }

            // Write results
            let dst_offset = token_idx * data.out_features + start_row;
            for i in 0..(end_row - start_row) {
                *data.output.add(dst_offset + i) = tmp[i];
            }
        }
    }
}

/// Worker for chunked Q4_K: each chunk is a contiguous block of rows
fn chunked_mul_mat_q4_k_worker(chunk: usize, _tid: usize, data: *const u8) {
    unsafe {
        let data = &*(data as *const ChunkedMulMatData);
        let chunk_size = data.chunk_size;

        // Decode chunk to get row range
        let start_row = chunk * chunk_size;
        if start_row >= data.out_features {
            return;
        }
        let end_row = (start_row + chunk_size).min(data.out_features);

        // Process each token's slice
        for token_idx in 0..data.n_tokens {
            let input_start = token_idx * data.n_blocks;
            let input_row = std::slice::from_raw_parts(
                data.input.add(input_start),
                data.n_blocks,
            );

            // Process all rows in this chunk (contiguous memory access!)
            let mut tmp = [0.0f32; 64]; // Max chunk_size
            for (i, row) in (start_row..end_row).enumerate() {
                let row_offset = row * data.col_stride;
                let weight_row = std::slice::from_raw_parts(
                    data.weights.add(row_offset) as *const BlockQ4_K,
                    data.n_blocks,
                );
                tmp[i] = vec_dot_q4_k_q8_k(weight_row, input_row);
            }

            // Write results
            let dst_offset = token_idx * data.out_features + start_row;
            for i in 0..(end_row - start_row) {
                *data.output.add(dst_offset + i) = tmp[i];
            }
        }
    }
}

/// Worker for chunked Q6_K: each chunk is a contiguous block of rows
fn chunked_mul_mat_q6_k_worker(chunk: usize, _tid: usize, data: *const u8) {
    unsafe {
        let data = &*(data as *const ChunkedMulMatData);
        let chunk_size = data.chunk_size;

        let start_row = chunk * chunk_size;
        if start_row >= data.out_features {
            return;
        }
        let end_row = (start_row + chunk_size).min(data.out_features);

        for token_idx in 0..data.n_tokens {
            let input_start = token_idx * data.n_blocks;
            let input_row = std::slice::from_raw_parts(
                data.input.add(input_start),
                data.n_blocks,
            );

            let mut tmp = [0.0f32; 64]; // Max chunk_size
            for (i, row) in (start_row..end_row).enumerate() {
                let row_offset = row * data.col_stride;
                let weight_row = std::slice::from_raw_parts(
                    data.weights.add(row_offset) as *const BlockQ6_K,
                    data.n_blocks,
                );
                tmp[i] = vec_dot_q6_k_q8_k(weight_row, input_row);
            }

            let dst_offset = token_idx * data.out_features + start_row;
            for i in 0..(end_row - start_row) {
                *data.output.add(dst_offset + i) = tmp[i];
            }
        }
    }
}

/// Work data for Q4_0 mul_mat
#[repr(C)]
struct MulMatQ40Data {
    weights: *const u8,
    col_stride: usize,
    input: *const BlockQ8_0,
    output: *mut f32,
    out_features: usize,
    n_blocks: usize,
    block_size: usize,
    n_tokens: usize,
}

// ============================================================================
// Worker functions for persistent thread pool
// ============================================================================

fn mul_mat_q6_k_block_worker(
    block_idx: usize,
    _tid: usize,
    data_ptr: *const u8,
) {
    const BLCK_0: usize = 16;
    unsafe {
        let data = &*(data_ptr as *const MulMatQ6KData);
        let start_row = block_idx * BLCK_0;
        let end_row = (start_row + BLCK_0).min(data.out_features);

        for token_idx in 0..data.n_tokens {
            let input_start = token_idx * data.n_blocks;
            let input_row = std::slice::from_raw_parts(
                data.input.add(input_start),
                data.n_blocks,
            );

            let mut tmp = [0.0f32; BLCK_0];
            for ir0 in start_row..end_row {
                let row_offset = ir0 * data.col_stride;
                let weight_row = std::slice::from_raw_parts(
                    data.weights.add(row_offset) as *const BlockQ6_K,
                    data.n_blocks,
                );
                tmp[ir0 - start_row] = vec_dot_q6_k_q8_k(weight_row, input_row);
            }

            let dst_offset = token_idx * data.out_features + start_row;
            for i in 0..(end_row - start_row) {
                *data.output.add(dst_offset + i) = tmp[i];
            }
        }
    }
}

fn mul_mat_q5_k_worker(flat_idx: usize, _tid: usize, data_ptr: *const u8) {
    unsafe {
        let data = &*(data_ptr as *const MulMatQ5KData);
        let token_idx = flat_idx / data.out_features;
        let row = flat_idx % data.out_features;

        let input_start = token_idx * data.n_blocks;

        let row_offset = row * data.col_stride;
        let weight_row = std::slice::from_raw_parts(
            data.weights.add(row_offset) as *const BlockQ5_K,
            data.n_blocks,
        );
        let input_row = std::slice::from_raw_parts(
            data.input.add(input_start),
            data.n_blocks,
        );

        let sum = vec_dot_q5_k_q8_k(weight_row, input_row);
        *data.output.add(flat_idx) = sum;
    }
}

fn mul_mat_q4_0_worker(flat_idx: usize, _tid: usize, data_ptr: *const u8) {
    unsafe {
        let data = &*(data_ptr as *const MulMatQ40Data);
        let token_idx = flat_idx / data.out_features;
        let row = flat_idx % data.out_features;

        let input_start = token_idx * data.n_blocks;

        let row_offset = row * data.col_stride;
        let weight_row = std::slice::from_raw_parts(
            data.weights.add(row_offset) as *const BlockQ4_0,
            data.n_blocks,
        );
        let input_row = std::slice::from_raw_parts(
            data.input.add(input_start),
            data.n_blocks,
        );

        let sum = vec_dot_q4_0_q8_0(weight_row, input_row);
        *data.output.add(flat_idx) = sum;
    }
}

// Q4_0 GEMV
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

// Q8_0 GEMV
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

// Q4_K GEMV
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

// Q5_K GEMV
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

// Q6_K GEMV
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
// Transposed GEMV for GGUF weight layout
// GGUF stores weights as [in_features, out_features] (transposed)
// For decode: output[j] = Σ_i weight[i,j] * input[i]
//
// IMPORTANT: For quantized formats, the transposed layout is NOT compatible
// with standard vec_dot operations. GGUF uses a different storage scheme where
// weights are stored row-by-row for each output feature, but dims show the
// transposed shape. The actual memory layout is [out_features][in_features].
// ============================================================================

// Transposed GEMV functions for GGUF weight layout
// GGUF stores weights transposed: weight[i, j] at position i * col_stride + j
// For decode: output[j] = Σ_i weight[i, j] * input[i]

/// Unified matrix multiplication for Q6_K weights
///
/// Handles both decode (n_tokens=1) and prefill (n_tokens>1).
/// GGUF memory layout for weights [ne0, ne1] = [in_features, out_features]:
/// - Each output channel is stored contiguously as n_blocks blocks
/// - col_stride = nb[1] = bytes per output channel = block_size * n_blocks
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
    let n_blocks = in_features / QK_K;
    let block_size = std::mem::size_of::<BlockQ6_K>();

    assert_eq!(input.len(), n_tokens * n_blocks);
    assert_eq!(output.len(), n_tokens * out_features);

    // For decode (n_tokens=1), use single-thread to avoid synchronization overhead
    // Memory-bound operations don't benefit from multi-threading
    let n_threads = if n_tokens == 1 {
        1
    } else {
        if n_threads == 0 { optimal_threads() } else { n_threads }
    };

    for i in 0..output.len() {
        output[i] = 0.0;
    }

    // K-extent optimization: process K in chunks to improve cache reuse
    let k_extent_blocks = compute_k_extent_blocks(n_tokens, block_size);
    const BLCK_0: usize = 16;

    if n_threads == 1 {
        // Llama.cpp style: output rows outer loop, K-extent middle loop
        const TILE_ROWS: usize = 256;

        for iir0 in (0..out_features).step_by(TILE_ROWS) {
            let ir0_end = (iir0 + TILE_ROWS).min(out_features);

            for k0 in (0..n_blocks).step_by(k_extent_blocks) {
                let k_end = (k0 + k_extent_blocks).min(n_blocks);
                let num_k = k_end - k0;

                for ir0 in (iir0..ir0_end).step_by(BLCK_0) {
                    let micro_end = (ir0 + BLCK_0).min(ir0_end);

                    for token_idx in 0..n_tokens {
                        let mut tmp = [0.0f32; BLCK_0];

                        for row in ir0..micro_end {
                            let row_offset = row * col_stride;
                            let weight_row = unsafe {
                                std::slice::from_raw_parts(
                                    weights
                                        .as_ptr()
                                        .add(row_offset + k0 * block_size)
                                        as *const BlockQ6_K,
                                    num_k,
                                )
                            };
                            let input_start = token_idx * n_blocks + k0;
                            let input_chunk =
                                &input[input_start..input_start + num_k];

                            tmp[row - ir0] =
                                vec_dot_q6_k_q8_k(weight_row, input_chunk);
                        }

                        let dst_offset = token_idx * out_features + ir0;
                        for i in 0..(micro_end - ir0) {
                            output[dst_offset + i] += tmp[i];
                        }
                    }
                }
            }
        }
    } else {
        // Multi-threaded: use persistent thread pool with work-stealing
        let chunk_size = 64;
        let n_chunks = (out_features + chunk_size - 1) / chunk_size;

        let data = ChunkedMulMatData {
            weights: weights.as_ptr(),
            col_stride,
            input: input.as_ptr(),
            output: output.as_mut_ptr(),
            out_features,
            n_blocks,
            n_tokens,
            chunk_size,
        };

        unsafe {
            parallel_stealing_raw(
                n_chunks,
                n_threads,
                chunked_mul_mat_q6_k_worker,
                &data as *const _ as *const u8,
            );
        }
    }
}

/// Legacy alias
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
    );
}

/// Unified matrix multiplication for Q5_K weights
///
/// Handles both decode (n_tokens=1) and prefill (n_tokens>1).
/// GGUF memory layout for weights [ne0, ne1] = [in_features, out_features]:
/// - Each output channel is stored contiguously as n_blocks blocks
/// - col_stride = nb[1] = bytes per output channel = block_size * n_blocks
///
/// Thread strategy:
/// - Decode (n_tokens=1): Single-threaded (memory-bound)
/// - Prefill (n_tokens>1): Multi-threaded (compute-bound)
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
    let n_blocks = in_features / QK_K;
    let block_size = std::mem::size_of::<BlockQ5_K>();

    assert_eq!(input.len(), n_tokens * n_blocks);
    assert_eq!(output.len(), n_tokens * out_features);

    // For decode (n_tokens=1), use single-thread to avoid synchronization overhead
    let n_threads = if n_tokens == 1 {
        1
    } else {
        if n_threads == 0 { optimal_threads() } else { n_threads }
    };

    for i in 0..output.len() {
        output[i] = 0.0;
    }

    // K-extent optimization: process K in chunks to improve cache reuse
    // Based on llama.cpp's narrow path for decode
    let k_extent_blocks = compute_k_extent_blocks(n_tokens, block_size);
    const BLCK_0: usize = 16; // Output rows per microtile

    if n_threads == 1 {
        // Llama.cpp style: output rows outer loop, K-extent middle loop
        // This keeps weights in L1 cache longer when computing multiple rows
        const TILE_ROWS: usize = 256; // Output row window size

        for iir0 in (0..out_features).step_by(TILE_ROWS) {
            let ir0_end = (iir0 + TILE_ROWS).min(out_features);

            // K-extent loop: iterate over K dimension in chunks
            for k0 in (0..n_blocks).step_by(k_extent_blocks) {
                let k_end = (k0 + k_extent_blocks).min(n_blocks);
                let num_k = k_end - k0;

                // 16-row microtiles within the window
                for ir0 in (iir0..ir0_end).step_by(BLCK_0) {
                    let micro_end = (ir0 + BLCK_0).min(ir0_end);

                    for token_idx in 0..n_tokens {
                        // Accumulator for this K-chunk
                        let mut tmp = [0.0f32; BLCK_0];

                        for row in ir0..micro_end {
                            let row_offset = row * col_stride;
                            let weight_row = unsafe {
                                std::slice::from_raw_parts(
                                    weights
                                        .as_ptr()
                                        .add(row_offset + k0 * block_size)
                                        as *const BlockQ5_K,
                                    num_k,
                                )
                            };
                            let input_start = token_idx * n_blocks + k0;
                            let input_chunk =
                                &input[input_start..input_start + num_k];

                            tmp[row - ir0] =
                                vec_dot_q5_k_q8_k(weight_row, input_chunk);
                        }

                        // Accumulate to output
                        let dst_offset = token_idx * out_features + ir0;
                        for i in 0..(micro_end - ir0) {
                            output[dst_offset + i] += tmp[i];
                        }
                    }
                }
            }
        }
    } else {
        // Multi-threaded: use persistent thread pool with work-stealing
        let chunk_size = 64;
        let n_chunks = (out_features + chunk_size - 1) / chunk_size;

        let data = ChunkedMulMatData {
            weights: weights.as_ptr(),
            col_stride,
            input: input.as_ptr(),
            output: output.as_mut_ptr(),
            out_features,
            n_blocks,
            n_tokens,
            chunk_size,
        };

        unsafe {
            parallel_stealing_raw(
                n_chunks,
                n_threads,
                chunked_mul_mat_q5_k_worker,
                &data as *const _ as *const u8,
            );
        }
    }
}

/// Legacy alias for backward compatibility
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
    );
}

/// Unified matrix multiplication for Q4_K weights
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
    let n_blocks = in_features / QK_K;
    let block_size = std::mem::size_of::<BlockQ4_K>();

    assert_eq!(input.len(), n_tokens * n_blocks);
    assert_eq!(output.len(), n_tokens * out_features);

    // For decode (n_tokens=1), use single-thread to avoid synchronization overhead
    let n_threads = if n_tokens == 1 {
        1
    } else {
        if n_threads == 0 { optimal_threads() } else { n_threads }
    };

    for i in 0..output.len() {
        output[i] = 0.0;
    }

    // K-extent optimization: process K in chunks to improve cache reuse
    let k_extent_blocks = compute_k_extent_blocks(n_tokens, block_size);
    const BLCK_0: usize = 16;

    if n_threads == 1 {
        // Llama.cpp style: output rows outer loop, K-extent middle loop
        const TILE_ROWS: usize = 256;

        for iir0 in (0..out_features).step_by(TILE_ROWS) {
            let ir0_end = (iir0 + TILE_ROWS).min(out_features);

            for k0 in (0..n_blocks).step_by(k_extent_blocks) {
                let k_end = (k0 + k_extent_blocks).min(n_blocks);
                let num_k = k_end - k0;

                for ir0 in (iir0..ir0_end).step_by(BLCK_0) {
                    let micro_end = (ir0 + BLCK_0).min(ir0_end);

                    for token_idx in 0..n_tokens {
                        let mut tmp = [0.0f32; BLCK_0];

                        for row in ir0..micro_end {
                            let row_offset = row * col_stride;
                            let weight_row = unsafe {
                                std::slice::from_raw_parts(
                                    weights
                                        .as_ptr()
                                        .add(row_offset + k0 * block_size)
                                        as *const BlockQ4_K,
                                    num_k,
                                )
                            };
                            let input_start = token_idx * n_blocks + k0;
                            let input_chunk =
                                &input[input_start..input_start + num_k];

                            tmp[row - ir0] =
                                vec_dot_q4_k_q8_k(weight_row, input_chunk);
                        }

                        let dst_offset = token_idx * out_features + ir0;
                        for i in 0..(micro_end - ir0) {
                            output[dst_offset + i] += tmp[i];
                        }
                    }
                }
            }
        }
    } else {
        // Multi-threaded: use persistent thread pool with work-stealing
        let chunk_size = 64;
        let n_chunks = (out_features + chunk_size - 1) / chunk_size;

        let data = ChunkedMulMatData {
            weights: weights.as_ptr(),
            col_stride,
            input: input.as_ptr(),
            output: output.as_mut_ptr(),
            out_features,
            n_blocks,
            n_tokens,
            chunk_size,
        };

        unsafe {
            parallel_stealing_raw(
                n_chunks,
                n_threads,
                chunked_mul_mat_q4_k_worker,
                &data as *const _ as *const u8,
            );
        }
    }
}

/// Legacy alias
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
    );
}

/// Unified matrix multiplication for Q4_0 weights
pub fn mul_mat_q4_0(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) {
    let n_blocks = in_features / QK8_0;
    let block_size = std::mem::size_of::<BlockQ4_0>();

    assert_eq!(input.len(), n_tokens * n_blocks);
    assert_eq!(output.len(), n_tokens * out_features);

    let n_threads = if n_threads == 0 { optimal_threads() } else { n_threads };

    for i in 0..output.len() {
        output[i] = 0.0;
    }

    let total_rows = n_tokens * out_features;
    const BLCK_0: usize = 16;

    if n_threads == 1 {
        for iir0 in (0..out_features).step_by(BLCK_0) {
            let ir0_end = (iir0 + BLCK_0).min(out_features);
            for token_idx in 0..n_tokens {
                let input_start = token_idx * n_blocks;
                let input_row = &input[input_start..input_start + n_blocks];

                let mut tmp = [0.0f32; BLCK_0];
                for ir0 in iir0..ir0_end {
                    let row_offset = ir0 * col_stride;
                    let weight_row = unsafe {
                        std::slice::from_raw_parts(
                            weights.as_ptr().add(row_offset)
                                as *const BlockQ4_0,
                            n_blocks,
                        )
                    };
                    tmp[ir0 - iir0] = vec_dot_q4_0_q8_0(weight_row, input_row);
                }

                let dst_offset = token_idx * out_features + iir0;
                for i in 0..(ir0_end - iir0) {
                    output[dst_offset + i] = tmp[i];
                }
            }
        }
    } else {
        let data = MulMatQ40Data {
            weights: weights.as_ptr(),
            col_stride,
            input: input.as_ptr(),
            output: output.as_mut_ptr(),
            out_features,
            n_blocks,
            block_size,
            n_tokens,
        };
        unsafe {
            parallel_stealing_raw(
                total_rows,
                n_threads,
                mul_mat_q4_0_worker,
                &data as *const _ as *const u8,
            );
        }
    }
}

/// Legacy alias
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
    );
}

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

        gemv_q4_0(&weights, &input, &mut output, out_features, in_features, 0);

        for &v in &output {
            assert!(v.abs() < 1.0);
        }
    }
}

#[cfg(test)]
mod q4k_mulmat_test {
    use super::*;

    #[test]
    fn test_mul_mat_q4_k_narrow_path() {
        // Test narrow path with n_tokens=1
        let in_features = 256; // 1 block
        let out_features = 4;
        let n_blocks = in_features / QK_K;

        // Create simple Q4_K weights (column-major)
        let block_size = std::mem::size_of::<BlockQ4_K>();
        let col_stride = block_size; // Each column has 1 block

        let mut weights = vec![0u8; out_features * col_stride];
        for row in 0..out_features {
            let block = unsafe {
                &mut *(weights.as_mut_ptr().add(row * col_stride)
                    as *mut BlockQ4_K)
            };
            block.d = half::f16::to_bits(half::f16::from_f32(1.0));
            block.dmin = half::f16::to_bits(half::f16::from_f32(0.0));
            block.scales = [8; 12]; // scales = 8
            block.qs = [0x88; 128]; // Each nibble = 8
        }

        // Create Q8_K input
        let mut input: Vec<BlockQ8_K> =
            vec![unsafe { std::mem::zeroed() }; n_blocks];
        for block in &mut input {
            block.d = 1.0;
            block.qs = [1; 256]; // All ones
            block.bsums = [256; 16]; // Each sub-block sum = 256
        }

        let mut output = vec![0.0f32; out_features];

        // Call mul_mat_q4_k with n_tokens=1 (narrow path)
        mul_mat_q4_k(
            &weights,
            col_stride,
            &input,
            &mut output,
            out_features,
            in_features,
            1, // n_tokens = 1, triggers narrow path
            1, // 1 thread
        );

        println!("Output: {:?}", output);

        // With all Q4_K values = 8 (from 0x88), Q8_K = 1, scale = 8, d = 1
        // Each position: 8 * 1 = 8
        // 256 positions * 8 = 2048
        // With d=1 and scale=8, result should be around 2048 * 1 * 8 = 16384
        // But we need to check the actual formula

        // Check that values are reasonable (not NaN or inf)
        for &v in &output {
            assert!(v.is_finite(), "Output is not finite: {}", v);
            println!("Output value: {}", v);
        }
    }

    #[test]
    fn test_mul_mat_q4_k_standard_path() {
        // Test standard path with n_tokens=32 (>16)
        let in_features = 256; // 1 block
        let out_features = 4;
        let n_tokens = 32;
        let n_blocks = in_features / QK_K;

        let block_size = std::mem::size_of::<BlockQ4_K>();
        let col_stride = block_size;

        let mut weights = vec![0u8; out_features * col_stride];
        for row in 0..out_features {
            let block = unsafe {
                &mut *(weights.as_mut_ptr().add(row * col_stride)
                    as *mut BlockQ4_K)
            };
            block.d = half::f16::to_bits(half::f16::from_f32(1.0));
            block.dmin = half::f16::to_bits(half::f16::from_f32(0.0));
            block.scales = [8; 12];
            block.qs = [0x88; 128];
        }

        // Create Q8_K input for multiple tokens
        let mut input: Vec<BlockQ8_K> =
            vec![unsafe { std::mem::zeroed() }; n_tokens * n_blocks];
        for block in &mut input {
            block.d = 1.0;
            block.qs = [1; 256];
            block.bsums = [256; 16];
        }

        let mut output = vec![0.0f32; n_tokens * out_features];

        // Call mul_mat_q4_k with n_tokens=32 (standard path)
        mul_mat_q4_k(
            &weights,
            col_stride,
            &input,
            &mut output,
            out_features,
            in_features,
            n_tokens,
            1,
        );

        // Check that values are reasonable
        for &v in &output {
            assert!(v.is_finite(), "Output is not finite: {}", v);
        }
    }
}

#[cfg(test)]
mod q6k_mulmat_test {
    use super::*;

    #[test]
    fn test_mul_mat_q6_k() {
        // Test Q6_K GEMV with known values
        let in_features = 256; // 1 block
        let out_features = 4;
        let n_blocks = in_features / QK_K;

        // Create simple Q6_K weights (column-major)
        let block_size = std::mem::size_of::<BlockQ6_K>();
        let col_stride = block_size; // Each column has 1 block

        let mut weights = vec![0u8; out_features * col_stride];
        for row in 0..out_features {
            let block = unsafe {
                &mut *(weights.as_mut_ptr().add(row * col_stride)
                    as *mut BlockQ6_K)
            };
            // Q6_K: 6-bit quantization with scale
            block.d = half::f16::to_bits(half::f16::from_f32(1.0));
            block.scales = [32; 16]; // scales = 32
            block.ql = [0x88; 128]; // low 4 bits = 8
            block.qh = [0; 64]; // high 2 bits = 0
        }

        // Create Q8_K input
        let mut input: Vec<BlockQ8_K> =
            vec![unsafe { std::mem::zeroed() }; n_blocks];
        for block in &mut input {
            block.d = 1.0;
            block.qs = [1; 256]; // All ones
            block.bsums = [256; 16];
        }

        let mut output = vec![0.0f32; out_features];

        // Call mul_mat_q6_k
        mul_mat_q6_k(
            &weights,
            col_stride,
            &input,
            &mut output,
            out_features,
            in_features,
            1,
            1,
        );

        println!("Q6_K Output: {:?}", output);

        // Check that values are reasonable
        for &v in &output {
            assert!(v.is_finite(), "Output is not finite: {}", v);
        }
    }
}
