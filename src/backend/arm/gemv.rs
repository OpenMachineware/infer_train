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
use crate::backend::thread_pool::parallel_stealing_raw;

pub const QK4_0: usize = 32;

// Block-tiling size from llama.cpp
// Use 64 for decode to reduce atomic operation overhead
// llama.cpp uses chunk_size=64 when nr1==1 (decode mode)
const BLOCK_ROWS: usize = 64;

// Block-tiling for prefill: 16 tokens × 16 rows
// Weight is reused 16 times in L1 cache
const BLK_0: usize = 16; // rows per block
const BLK_1: usize = 16; // tokens per block

// ============================================================================
// Work data structures for thread pool
// ============================================================================

#[repr(C)]
struct GemvWorkData {
    weight_ptr: *const u8,
    row_stride: usize,
    input_ptr: *const BlockQ8_K,
    input_len: usize,
    output_ptr: *mut f32,
    out_features: usize,
    nb: usize,
    vec_dot_fn: unsafe fn(*const u8, usize, *const BlockQ8_K, usize, *mut f32),
}

/// Generic work function for GEMV block processing
unsafe fn gemv_work_fn(block_idx: usize, _tid: usize, data: *const u8) {
    let work = &*(data as *const GemvWorkData);
    let row_start = block_idx * BLOCK_ROWS;
    let row_end = (row_start + BLOCK_ROWS).min(work.out_features);

    let input_slice =
        std::slice::from_raw_parts(work.input_ptr, work.input_len);
    let mut tmp = [0.0f32; BLOCK_ROWS];

    for i in row_start..row_end {
        (work.vec_dot_fn)(
            work.weight_ptr.add(i * work.row_stride),
            work.nb,
            input_slice.as_ptr(),
            input_slice.len(),
            &mut tmp[i - row_start],
        );
    }

    for i in row_start..row_end {
        *work.output_ptr.add(i) = tmp[i - row_start];
    }
}

// ============================================================================
// Vec dot wrappers for type erasure
// ============================================================================

unsafe fn vec_dot_q4_k_wrapper(
    row_ptr: *const u8,
    nb: usize,
    input_ptr: *const BlockQ8_K,
    _input_len: usize,
    out: *mut f32,
) {
    let row_blocks =
        std::slice::from_raw_parts(row_ptr as *const BlockQ4_K, nb);
    let input = std::slice::from_raw_parts(input_ptr, nb);
    *out = vec_dot_q4_k_q8_k(row_blocks, input);
}

unsafe fn vec_dot_q5_k_wrapper(
    row_ptr: *const u8,
    nb: usize,
    input_ptr: *const BlockQ8_K,
    _input_len: usize,
    out: *mut f32,
) {
    let row_blocks =
        std::slice::from_raw_parts(row_ptr as *const BlockQ5_K, nb);
    let input = std::slice::from_raw_parts(input_ptr, nb);
    *out = vec_dot_q5_k_q8_k(row_blocks, input);
}

unsafe fn vec_dot_q6_k_wrapper(
    row_ptr: *const u8,
    nb: usize,
    input_ptr: *const BlockQ8_K,
    _input_len: usize,
    out: *mut f32,
) {
    let row_blocks =
        std::slice::from_raw_parts(row_ptr as *const BlockQ6_K, nb);
    let input = std::slice::from_raw_parts(input_ptr, nb);
    *out = vec_dot_q6_k_q8_k(row_blocks, input);
}

unsafe fn vec_dot_iq4_xs_wrapper(
    row_ptr: *const u8,
    nb: usize,
    input_ptr: *const BlockQ8_K,
    _input_len: usize,
    out: *mut f32,
) {
    let row_blocks =
        std::slice::from_raw_parts(row_ptr as *const BlockIQ4XS, nb);
    let input = std::slice::from_raw_parts(input_ptr, nb);
    *out = vec_dot_iq4_xs_q8_k(row_blocks, input);
}

// ============================================================================
// Single-token GEMV (decode path) - using thread pool
// ============================================================================

/// GEMV for Q4_K with block-tiling using persistent thread pool
pub unsafe fn gemv_q4_k_transposed(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_threads: usize,
) {
    let nb = in_features / QK_K;
    let row_stride = if weight_stride > 0 {
        weight_stride
    } else {
        nb * std::mem::size_of::<BlockQ4_K>()
    };

    gemv_with_pool(
        weight_data,
        row_stride,
        input,
        output,
        out_features,
        nb,
        n_threads,
        vec_dot_q4_k_wrapper,
    );
}

/// GEMV for Q5_K with block-tiling using persistent thread pool
pub unsafe fn gemv_q5_k_transposed(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_threads: usize,
) {
    let nb = in_features / QK_K;
    let row_stride = if weight_stride > 0 {
        weight_stride
    } else {
        nb * std::mem::size_of::<BlockQ5_K>()
    };

    gemv_with_pool(
        weight_data,
        row_stride,
        input,
        output,
        out_features,
        nb,
        n_threads,
        vec_dot_q5_k_wrapper,
    );
}

/// GEMV for Q6_K with block-tiling using persistent thread pool
pub unsafe fn gemv_q6_k_transposed(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_threads: usize,
) {
    let nb = in_features / QK_K;
    let row_stride = if weight_stride > 0 {
        weight_stride
    } else {
        nb * std::mem::size_of::<BlockQ6_K>()
    };

    gemv_with_pool(
        weight_data,
        row_stride,
        input,
        output,
        out_features,
        nb,
        n_threads,
        vec_dot_q6_k_wrapper,
    );
}

/// GEMV for IQ4_XS with block-tiling using persistent thread pool
pub unsafe fn gemv_iq4_xs_transposed(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_threads: usize,
) {
    let nb = in_features / QK_K;
    let row_stride = if weight_stride > 0 {
        weight_stride
    } else {
        nb * std::mem::size_of::<BlockIQ4XS>()
    };

    gemv_with_pool(
        weight_data,
        row_stride,
        input,
        output,
        out_features,
        nb,
        n_threads,
        vec_dot_iq4_xs_wrapper,
    );
}

/// Generic GEMV using persistent thread pool
#[inline(always)]
unsafe fn gemv_with_pool(
    weight_data: &[u8],
    row_stride: usize,
    input: &[BlockQ8_K],
    output: &mut [f32],
    out_features: usize,
    nb: usize,
    n_threads: usize,
    vec_dot_fn: unsafe fn(*const u8, usize, *const BlockQ8_K, usize, *mut f32),
) {
    if n_threads > 1 && out_features >= BLOCK_ROWS * 2 {
        let work = GemvWorkData {
            weight_ptr: weight_data.as_ptr(),
            row_stride,
            input_ptr: input.as_ptr(),
            input_len: input.len(),
            output_ptr: output.as_mut_ptr(),
            out_features,
            nb,
            vec_dot_fn,
        };

        let n_blocks = (out_features + BLOCK_ROWS - 1) / BLOCK_ROWS;
        parallel_stealing_raw(
            n_blocks,
            n_threads,
            gemv_work_fn,
            &work as *const _ as *const u8,
        );
    } else {
        // Sequential processing
        for block_start in (0..out_features).step_by(BLOCK_ROWS) {
            let block_end = (block_start + BLOCK_ROWS).min(out_features);
            let mut tmp = [0.0f32; BLOCK_ROWS];

            for i in block_start..block_end {
                let row_ptr = weight_data.as_ptr().add(i * row_stride);
                (vec_dot_fn)(
                    row_ptr,
                    nb,
                    input.as_ptr(),
                    input.len(),
                    &mut tmp[i - block_start],
                );
            }

            for i in block_start..block_end {
                output[i] = tmp[i - block_start];
            }
        }
    }
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

/// Batched GEMV for Q4_K with proper block-tiling for L1 cache reuse
/// Based on llama.cpp's ggml_compute_forward_mul_mat_one_chunk
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

    // Block-tiling: 16 tokens × 16 rows
    // Weight is reused 16 times in L1 cache
    if n_tokens > 1 && n_threads > 1 {
        // Parallel block-tiling using persistent thread pool
        let n_blk0 = (out_features + BLK_0 - 1) / BLK_0; // row blocks
        let n_blk1 = (n_tokens + BLK_1 - 1) / BLK_1; // token blocks
        let total_blocks = n_blk0 * n_blk1;

        #[repr(C)]
        struct BlockWork {
            output_ptr: usize,
            weight_ptr: usize,
            input_ptr: usize,
            out_features: usize,
            n_tokens: usize,
            nb: usize,
            row_stride: usize,
            n_blk1: usize,
        }

        let work = BlockWork {
            output_ptr: output.as_mut_ptr() as usize,
            weight_ptr: weight_data.as_ptr() as usize,
            input_ptr: input.as_ptr() as usize,
            out_features,
            n_tokens,
            nb,
            row_stride,
            n_blk1,
        };

        unsafe fn block_work_fn(blk_idx: usize, _tid: usize, data: *const u8) {
            let work = &*(data as *const BlockWork);
            let output_ptr = work.output_ptr as *mut f32;
            let weight_ptr = work.weight_ptr as *const u8;
            let input_ptr = work.input_ptr as *const BlockQ8_K;

            let mut tmp = [0.0f32; BLK_0];

            let blk0 = blk_idx / work.n_blk1; // row block
            let blk1 = blk_idx % work.n_blk1; // token block

            let row_start = blk0 * BLK_0;
            let row_end = (row_start + BLK_0).min(work.out_features);
            let token_start = blk1 * BLK_1;
            let token_end = (token_start + BLK_1).min(work.n_tokens);

            for t in token_start..token_end {
                let input_slice = std::slice::from_raw_parts(
                    input_ptr.add(t * work.nb),
                    work.nb,
                );

                for i_row in row_start..row_end {
                    let row_blocks: &[BlockQ4_K] = std::slice::from_raw_parts(
                        weight_ptr.add(i_row * work.row_stride)
                            as *const BlockQ4_K,
                        work.nb,
                    );
                    tmp[i_row - row_start] =
                        vec_dot_q4_k_q8_k(row_blocks, input_slice);
                }

                for i_row in row_start..row_end {
                    *output_ptr.add(t * work.out_features + i_row) =
                        tmp[i_row - row_start];
                }
            }
        }

        unsafe {
            crate::backend::thread_pool::parallel_stealing_raw(
                total_blocks,
                n_threads,
                block_work_fn,
                &work as *const _ as *const u8,
            );
        }
    } else {
        // Sequential block-tiling
        let mut tmp = [0.0f32; BLK_0];

        for token_start in (0..n_tokens).step_by(BLK_1) {
            let token_end = (token_start + BLK_1).min(n_tokens);

            for row_start in (0..out_features).step_by(BLK_0) {
                let row_end = (row_start + BLK_0).min(out_features);

                for t in token_start..token_end {
                    let input_slice = &input[t * nb..(t + 1) * nb];

                    for i_row in row_start..row_end {
                        let row_blocks: &[BlockQ4_K] = unsafe {
                            std::slice::from_raw_parts(
                                weight_data.as_ptr().add(i_row * row_stride)
                                    as *const BlockQ4_K,
                                nb,
                            )
                        };
                        tmp[i_row - row_start] =
                            vec_dot_q4_k_q8_k(row_blocks, input_slice);
                    }

                    for i_row in row_start..row_end {
                        output[t * out_features + i_row] =
                            tmp[i_row - row_start];
                    }
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

    // Block-tiling: 16 tokens × 16 rows
    if n_tokens > 1 && n_threads > 1 {
        let n_blk0 = (out_features + BLK_0 - 1) / BLK_0;
        let n_blk1 = (n_tokens + BLK_1 - 1) / BLK_1;
        let total_blocks = n_blk0 * n_blk1;

        #[repr(C)]
        struct BlockWork {
            output_ptr: usize,
            weight_ptr: usize,
            input_ptr: usize,
            out_features: usize,
            n_tokens: usize,
            nb: usize,
            row_stride: usize,
            n_blk1: usize,
        }

        let work = BlockWork {
            output_ptr: output.as_mut_ptr() as usize,
            weight_ptr: weight_data.as_ptr() as usize,
            input_ptr: input.as_ptr() as usize,
            out_features,
            n_tokens,
            nb,
            row_stride,
            n_blk1,
        };

        unsafe fn block_work_fn(blk_idx: usize, _tid: usize, data: *const u8) {
            let work = &*(data as *const BlockWork);
            let output_ptr = work.output_ptr as *mut f32;
            let weight_ptr = work.weight_ptr as *const u8;
            let input_ptr = work.input_ptr as *const BlockQ8_K;

            let mut tmp = [0.0f32; BLK_0];

            let blk0 = blk_idx / work.n_blk1;
            let blk1 = blk_idx % work.n_blk1;

            let row_start = blk0 * BLK_0;
            let row_end = (row_start + BLK_0).min(work.out_features);
            let token_start = blk1 * BLK_1;
            let token_end = (token_start + BLK_1).min(work.n_tokens);

            for t in token_start..token_end {
                let input_slice = std::slice::from_raw_parts(
                    input_ptr.add(t * work.nb),
                    work.nb,
                );

                for i_row in row_start..row_end {
                    let row_blocks: &[BlockQ5_K] = std::slice::from_raw_parts(
                        weight_ptr.add(i_row * work.row_stride)
                            as *const BlockQ5_K,
                        work.nb,
                    );
                    tmp[i_row - row_start] =
                        vec_dot_q5_k_q8_k(row_blocks, input_slice);
                }

                for i_row in row_start..row_end {
                    *output_ptr.add(t * work.out_features + i_row) =
                        tmp[i_row - row_start];
                }
            }
        }

        unsafe {
            crate::backend::thread_pool::parallel_stealing_raw(
                total_blocks,
                n_threads,
                block_work_fn,
                &work as *const _ as *const u8,
            );
        }
    } else {
        let mut tmp = [0.0f32; BLK_0];

        for token_start in (0..n_tokens).step_by(BLK_1) {
            let token_end = (token_start + BLK_1).min(n_tokens);

            for row_start in (0..out_features).step_by(BLK_0) {
                let row_end = (row_start + BLK_0).min(out_features);

                for t in token_start..token_end {
                    let input_slice = &input[t * nb..(t + 1) * nb];

                    for i_row in row_start..row_end {
                        let row_blocks: &[BlockQ5_K] = unsafe {
                            std::slice::from_raw_parts(
                                weight_data.as_ptr().add(i_row * row_stride)
                                    as *const BlockQ5_K,
                                nb,
                            )
                        };
                        tmp[i_row - row_start] =
                            vec_dot_q5_k_q8_k(row_blocks, input_slice);
                    }

                    for i_row in row_start..row_end {
                        output[t * out_features + i_row] =
                            tmp[i_row - row_start];
                    }
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

    // Block-tiling: 16 tokens × 16 rows
    if n_tokens > 1 && n_threads > 1 {
        let n_blk0 = (out_features + BLK_0 - 1) / BLK_0;
        let n_blk1 = (n_tokens + BLK_1 - 1) / BLK_1;
        let total_blocks = n_blk0 * n_blk1;

        #[repr(C)]
        struct BlockWork {
            output_ptr: usize,
            weight_ptr: usize,
            input_ptr: usize,
            out_features: usize,
            n_tokens: usize,
            nb: usize,
            row_stride: usize,
            n_blk1: usize,
        }

        let work = BlockWork {
            output_ptr: output.as_mut_ptr() as usize,
            weight_ptr: weight_data.as_ptr() as usize,
            input_ptr: input.as_ptr() as usize,
            out_features,
            n_tokens,
            nb,
            row_stride,
            n_blk1,
        };

        unsafe fn block_work_fn(blk_idx: usize, _tid: usize, data: *const u8) {
            let work = &*(data as *const BlockWork);
            let output_ptr = work.output_ptr as *mut f32;
            let weight_ptr = work.weight_ptr as *const u8;
            let input_ptr = work.input_ptr as *const BlockQ8_K;

            let mut tmp = [0.0f32; BLK_0];

            let blk0 = blk_idx / work.n_blk1;
            let blk1 = blk_idx % work.n_blk1;

            let row_start = blk0 * BLK_0;
            let row_end = (row_start + BLK_0).min(work.out_features);
            let token_start = blk1 * BLK_1;
            let token_end = (token_start + BLK_1).min(work.n_tokens);

            for t in token_start..token_end {
                let input_slice = std::slice::from_raw_parts(
                    input_ptr.add(t * work.nb),
                    work.nb,
                );

                for i_row in row_start..row_end {
                    let row_blocks: &[BlockQ6_K] = std::slice::from_raw_parts(
                        weight_ptr.add(i_row * work.row_stride)
                            as *const BlockQ6_K,
                        work.nb,
                    );
                    tmp[i_row - row_start] =
                        vec_dot_q6_k_q8_k(row_blocks, input_slice);
                }

                for i_row in row_start..row_end {
                    *output_ptr.add(t * work.out_features + i_row) =
                        tmp[i_row - row_start];
                }
            }
        }

        unsafe {
            crate::backend::thread_pool::parallel_stealing_raw(
                total_blocks,
                n_threads,
                block_work_fn,
                &work as *const _ as *const u8,
            );
        }
    } else {
        let mut tmp = [0.0f32; BLK_0];

        for token_start in (0..n_tokens).step_by(BLK_1) {
            let token_end = (token_start + BLK_1).min(n_tokens);

            for row_start in (0..out_features).step_by(BLK_0) {
                let row_end = (row_start + BLK_0).min(out_features);

                for t in token_start..token_end {
                    let input_slice = &input[t * nb..(t + 1) * nb];

                    for i_row in row_start..row_end {
                        let row_blocks: &[BlockQ6_K] = unsafe {
                            std::slice::from_raw_parts(
                                weight_data.as_ptr().add(i_row * row_stride)
                                    as *const BlockQ6_K,
                                nb,
                            )
                        };
                        tmp[i_row - row_start] =
                            vec_dot_q6_k_q8_k(row_blocks, input_slice);
                    }

                    for i_row in row_start..row_end {
                        output[t * out_features + i_row] =
                            tmp[i_row - row_start];
                    }
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

    // Block-tiling: 16 tokens × 16 rows
    if n_tokens > 1 && n_threads > 1 {
        let n_blk0 = (out_features + BLK_0 - 1) / BLK_0;
        let n_blk1 = (n_tokens + BLK_1 - 1) / BLK_1;
        let total_blocks = n_blk0 * n_blk1;

        #[repr(C)]
        struct BlockWork {
            output_ptr: usize,
            weight_ptr: usize,
            input_ptr: usize,
            out_features: usize,
            n_tokens: usize,
            nb: usize,
            row_stride: usize,
            n_blk1: usize,
        }

        let work = BlockWork {
            output_ptr: output.as_mut_ptr() as usize,
            weight_ptr: weight_data.as_ptr() as usize,
            input_ptr: input.as_ptr() as usize,
            out_features,
            n_tokens,
            nb,
            row_stride,
            n_blk1,
        };

        unsafe fn block_work_fn(blk_idx: usize, _tid: usize, data: *const u8) {
            let work = &*(data as *const BlockWork);
            let output_ptr = work.output_ptr as *mut f32;
            let weight_ptr = work.weight_ptr as *const u8;
            let input_ptr = work.input_ptr as *const BlockQ8_K;

            let mut tmp = [0.0f32; BLK_0];

            let blk0 = blk_idx / work.n_blk1;
            let blk1 = blk_idx % work.n_blk1;

            let row_start = blk0 * BLK_0;
            let row_end = (row_start + BLK_0).min(work.out_features);
            let token_start = blk1 * BLK_1;
            let token_end = (token_start + BLK_1).min(work.n_tokens);

            for t in token_start..token_end {
                let input_slice = std::slice::from_raw_parts(
                    input_ptr.add(t * work.nb),
                    work.nb,
                );

                for i_row in row_start..row_end {
                    let row_blocks: &[BlockIQ4XS] = std::slice::from_raw_parts(
                        weight_ptr.add(i_row * work.row_stride)
                            as *const BlockIQ4XS,
                        work.nb,
                    );
                    tmp[i_row - row_start] =
                        vec_dot_iq4_xs_q8_k(row_blocks, input_slice);
                }

                for i_row in row_start..row_end {
                    *output_ptr.add(t * work.out_features + i_row) =
                        tmp[i_row - row_start];
                }
            }
        }

        unsafe {
            crate::backend::thread_pool::parallel_stealing_raw(
                total_blocks,
                n_threads,
                block_work_fn,
                &work as *const _ as *const u8,
            );
        }
    } else {
        let mut tmp = [0.0f32; BLK_0];

        for token_start in (0..n_tokens).step_by(BLK_1) {
            let token_end = (token_start + BLK_1).min(n_tokens);

            for row_start in (0..out_features).step_by(BLK_0) {
                let row_end = (row_start + BLK_0).min(out_features);

                for t in token_start..token_end {
                    let input_slice = &input[t * nb..(t + 1) * nb];

                    for i_row in row_start..row_end {
                        let row_blocks: &[BlockIQ4XS] = unsafe {
                            std::slice::from_raw_parts(
                                weight_data.as_ptr().add(i_row * row_stride)
                                    as *const BlockIQ4XS,
                                nb,
                            )
                        };
                        tmp[i_row - row_start] =
                            vec_dot_iq4_xs_q8_k(row_blocks, input_slice);
                    }

                    for i_row in row_start..row_end {
                        output[t * out_features + i_row] =
                            tmp[i_row - row_start];
                    }
                }
            }
        }
    }
}

/// Batched GEMV for Q4_0 with block-tiling
pub fn mul_mat_q4_0(
    weight_data: &[u8],
    weight_stride: usize,
    input: &[BlockQ8_0],
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) {
    let nb = in_features / QK4_0;
    let row_stride = weight_stride;

    // Block-tiling: 16 tokens × 16 rows
    if n_tokens > 1 && n_threads > 1 {
        let n_blk0 = (out_features + BLK_0 - 1) / BLK_0;
        let n_blk1 = (n_tokens + BLK_1 - 1) / BLK_1;
        let total_blocks = n_blk0 * n_blk1;

        #[repr(C)]
        struct BlockWork {
            output_ptr: usize,
            weight_ptr: usize,
            input_ptr: usize,
            out_features: usize,
            n_tokens: usize,
            nb: usize,
            row_stride: usize,
            n_blk1: usize,
        }

        let work = BlockWork {
            output_ptr: output.as_mut_ptr() as usize,
            weight_ptr: weight_data.as_ptr() as usize,
            input_ptr: input.as_ptr() as usize,
            out_features,
            n_tokens,
            nb,
            row_stride,
            n_blk1,
        };

        unsafe fn block_work_fn(blk_idx: usize, _tid: usize, data: *const u8) {
            let work = &*(data as *const BlockWork);
            let output_ptr = work.output_ptr as *mut f32;
            let weight_ptr = work.weight_ptr as *const u8;
            let input_ptr = work.input_ptr as *const BlockQ8_0;

            let mut tmp = [0.0f32; BLK_0];

            let blk0 = blk_idx / work.n_blk1;
            let blk1 = blk_idx % work.n_blk1;

            let row_start = blk0 * BLK_0;
            let row_end = (row_start + BLK_0).min(work.out_features);
            let token_start = blk1 * BLK_1;
            let token_end = (token_start + BLK_1).min(work.n_tokens);

            for t in token_start..token_end {
                let input_slice = std::slice::from_raw_parts(
                    input_ptr.add(t * work.nb),
                    work.nb,
                );

                for i_row in row_start..row_end {
                    let row_blocks: &[BlockQ4_0] = std::slice::from_raw_parts(
                        weight_ptr.add(i_row * work.row_stride)
                            as *const BlockQ4_0,
                        work.nb,
                    );
                    tmp[i_row - row_start] =
                        vec_dot_q4_0_q8_0(row_blocks, input_slice);
                }

                for i_row in row_start..row_end {
                    *output_ptr.add(t * work.out_features + i_row) =
                        tmp[i_row - row_start];
                }
            }
        }

        unsafe {
            crate::backend::thread_pool::parallel_stealing_raw(
                total_blocks,
                n_threads,
                block_work_fn,
                &work as *const _ as *const u8,
            );
        }
    } else {
        let mut tmp = [0.0f32; BLK_0];

        for token_start in (0..n_tokens).step_by(BLK_1) {
            let token_end = (token_start + BLK_1).min(n_tokens);

            for row_start in (0..out_features).step_by(BLK_0) {
                let row_end = (row_start + BLK_0).min(out_features);

                for t in token_start..token_end {
                    let input_slice = &input[t * nb..(t + 1) * nb];

                    for i_row in row_start..row_end {
                        let row_blocks: &[BlockQ4_0] = unsafe {
                            std::slice::from_raw_parts(
                                weight_data.as_ptr().add(i_row * row_stride)
                                    as *const BlockQ4_0,
                                nb,
                            )
                        };
                        tmp[i_row - row_start] =
                            vec_dot_q4_0_q8_0(row_blocks, input_slice);
                    }

                    for i_row in row_start..row_end {
                        output[t * out_features + i_row] =
                            tmp[i_row - row_start];
                    }
                }
            }
        }
    }
}
