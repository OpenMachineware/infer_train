// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Matrix-vector multiplication (GEMV) for decode

use super::vec_dot::*;
use crate::backend::thread_pool::{optimal_threads, parallel_stealing_raw};
use crate::types::*;

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

/// Work data for Q4_K mul_mat
#[repr(C)]
struct MulMatQ4KData {
    weights: *const u8,
    col_stride: usize,
    input: *const BlockQ8_K,
    output: *mut f32,
    out_features: usize,
    n_blocks: usize,
    block_size: usize,
    n_tokens: usize,
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

fn mul_mat_q6_k_worker(flat_idx: usize, _tid: usize, data_ptr: *const u8) {
    unsafe {
        let data = &*(data_ptr as *const MulMatQ6KData);
        let token_idx = flat_idx / data.out_features;
        let row = flat_idx % data.out_features;

        let mut sum = 0.0f32;
        let input_start = token_idx * data.n_blocks;
        for block_idx in 0..data.n_blocks {
            let input_block = &*data.input.add(input_start + block_idx);
            let offset = block_idx * data.col_stride + row * data.block_size;
            let weight_block: &BlockQ6_K =
                &*(data.weights.add(offset) as *const BlockQ6_K);
            sum += vec_dot_q6_k_q8_k_single(weight_block, input_block);
        }
        *data.output.add(flat_idx) = sum;
    }
}

fn mul_mat_q5_k_worker(flat_idx: usize, _tid: usize, data_ptr: *const u8) {
    unsafe {
        let data = &*(data_ptr as *const MulMatQ5KData);
        let token_idx = flat_idx / data.out_features;
        let row = flat_idx % data.out_features;

        let mut sum = 0.0f32;
        let input_start = token_idx * data.n_blocks;
        for block_idx in 0..data.n_blocks {
            let input_block = &*data.input.add(input_start + block_idx);
            let offset = block_idx * data.col_stride + row * data.block_size;
            let weight_block: &BlockQ5_K =
                &*(data.weights.add(offset) as *const BlockQ5_K);
            sum += vec_dot_q5_k_q8_k_single(weight_block, input_block);
        }
        *data.output.add(flat_idx) = sum;
    }
}

fn mul_mat_q4_k_worker(flat_idx: usize, _tid: usize, data_ptr: *const u8) {
    unsafe {
        let data = &*(data_ptr as *const MulMatQ4KData);
        let token_idx = flat_idx / data.out_features;
        let row = flat_idx % data.out_features;

        let mut sum = 0.0f32;
        let input_start = token_idx * data.n_blocks;
        for block_idx in 0..data.n_blocks {
            let input_block = &*data.input.add(input_start + block_idx);
            let offset = block_idx * data.col_stride + row * data.block_size;
            let weight_block: &BlockQ4_K =
                &*(data.weights.add(offset) as *const BlockQ4_K);
            sum += vec_dot_q4_k_q8_k_single(weight_block, input_block);
        }
        *data.output.add(flat_idx) = sum;
    }
}

fn mul_mat_q4_0_worker(flat_idx: usize, _tid: usize, data_ptr: *const u8) {
    unsafe {
        let data = &*(data_ptr as *const MulMatQ40Data);
        let token_idx = flat_idx / data.out_features;
        let row = flat_idx % data.out_features;

        let mut sum = 0.0f32;
        let input_start = token_idx * data.n_blocks;
        for block_idx in 0..data.n_blocks {
            let input_block = &*data.input.add(input_start + block_idx);
            let offset = block_idx * data.col_stride + row * data.block_size;
            let weight_block: &BlockQ4_0 =
                &*(data.weights.add(offset) as *const BlockQ4_0);
            sum += vec_dot_q4_0_q8_0_single(weight_block, input_block);
        }
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

// Transposed GEMV functions for GGUF weight layout
// GGUF stores weights transposed: weight[i, j] at position i * col_stride + j
// For decode: output[j] = Σ_i weight[i, j] * input[i]

/// Unified matrix multiplication for Q6_K weights
///
/// Handles both decode (n_tokens=1) and prefill (n_tokens>1).
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

    let n_threads = if n_threads == 0 {
        optimal_threads(out_features, n_tokens, in_features)
    } else {
        n_threads
    };

    // Clear output
    for i in 0..output.len() {
        output[i] = 0.0;
    }

    let total_rows = n_tokens * out_features;

    if n_threads == 1 {
        // Single-threaded path
        for flat_idx in 0..total_rows {
            let token_idx = flat_idx / out_features;
            let row = flat_idx % out_features;

            let mut sum = 0.0f32;
            let input_start = token_idx * n_blocks;
            for block_idx in 0..n_blocks {
                let input_block = &input[input_start + block_idx];
                let offset = block_idx * col_stride + row * block_size;
                let weight_block: &BlockQ6_K = unsafe {
                    &*(weights.as_ptr().add(offset) as *const BlockQ6_K)
                };
                sum += vec_dot_q6_k_q8_k_single(weight_block, input_block);
            }
            output[flat_idx] = sum;
        }
    } else {
        // Multi-threaded path using persistent thread pool
        let data = MulMatQ6KData {
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
                mul_mat_q6_k_worker,
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
/// GGUF memory layout for weights [ne0, ne1] = [out, in]:
/// - Weight blocks stored as: [in_blocks][out_rows]
/// - col_stride = bytes for all output rows per input block
///
/// Thread strategy:
/// - Decode (n_tokens=1): Single-threaded (memory-bound)
/// - Prefill (n_tokens>1): Multi-threaded (compute-bound)
pub fn mul_mat_q5_k(
    weights: &[u8],
    col_stride: usize,
    input: &[BlockQ8_K], // [n_tokens * n_blocks]
    output: &mut [f32],  // [n_tokens * out_features]
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) {
    let n_blocks = in_features / QK_K;
    let block_size = std::mem::size_of::<BlockQ5_K>();

    assert_eq!(input.len(), n_tokens * n_blocks);
    assert_eq!(output.len(), n_tokens * out_features);

    // Thread selection based on workload
    let n_threads = if n_threads == 0 {
        optimal_threads(out_features, n_tokens, in_features)
    } else {
        n_threads
    };

    // Clear output
    for i in 0..output.len() {
        output[i] = 0.0;
    }

    let total_rows = n_tokens * out_features;

    if n_threads == 1 {
        // Single-threaded path
        for flat_idx in 0..total_rows {
            let token_idx = flat_idx / out_features;
            let row = flat_idx % out_features;

            let mut sum = 0.0f32;
            let input_start = token_idx * n_blocks;
            for block_idx in 0..n_blocks {
                let input_block = &input[input_start + block_idx];
                let offset = block_idx * col_stride + row * block_size;
                let weight_block: &BlockQ5_K = unsafe {
                    &*(weights.as_ptr().add(offset) as *const BlockQ5_K)
                };
                sum += vec_dot_q5_k_q8_k_single(weight_block, input_block);
            }
            output[flat_idx] = sum;
        }
    } else {
        // Multi-threaded path using persistent thread pool
        let data = MulMatQ5KData {
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
                mul_mat_q5_k_worker,
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

    let n_threads = if n_threads == 0 {
        optimal_threads(out_features, n_tokens, in_features)
    } else {
        n_threads
    };

    for i in 0..output.len() {
        output[i] = 0.0;
    }

    let total_rows = n_tokens * out_features;

    if n_threads == 1 {
        // Single-threaded path
        for flat_idx in 0..total_rows {
            let token_idx = flat_idx / out_features;
            let row = flat_idx % out_features;

            let mut sum = 0.0f32;
            let input_start = token_idx * n_blocks;
            for block_idx in 0..n_blocks {
                let input_block = &input[input_start + block_idx];
                let offset = block_idx * col_stride + row * block_size;
                let weight_block: &BlockQ4_K = unsafe {
                    &*(weights.as_ptr().add(offset) as *const BlockQ4_K)
                };
                sum += vec_dot_q4_k_q8_k_single(weight_block, input_block);
            }
            output[flat_idx] = sum;
        }
    } else {
        // Multi-threaded path using persistent thread pool
        let data = MulMatQ4KData {
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
                mul_mat_q4_k_worker,
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

    let n_threads = if n_threads == 0 {
        optimal_threads(out_features, n_tokens, in_features)
    } else {
        n_threads
    };

    for i in 0..output.len() {
        output[i] = 0.0;
    }

    let total_rows = n_tokens * out_features;

    if n_threads == 1 {
        // Single-threaded path
        for flat_idx in 0..total_rows {
            let token_idx = flat_idx / out_features;
            let row = flat_idx % out_features;

            let mut sum = 0.0f32;
            let input_start = token_idx * n_blocks;
            for block_idx in 0..n_blocks {
                let input_block = &input[input_start + block_idx];
                let offset = block_idx * col_stride + row * block_size;
                let weight_block: &BlockQ4_0 = unsafe {
                    &*(weights.as_ptr().add(offset) as *const BlockQ4_0)
                };
                sum += vec_dot_q4_0_q8_0_single(weight_block, input_block);
            }
            output[flat_idx] = sum;
        }
    } else {
        // Multi-threaded path using persistent thread pool
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
