// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Copyright 2026 Jia Liu
//
// Section 7: Additional permission under GNU AGPL version 3 section 7
// If you modify this Program, or any covered work, by linking or combining
// it with OpenSSL (or a modified version of that library), containing parts
// covered by the terms of OpenSSL license, the licensors of this Program
// grant you additional permission to convey the resulting work.
// Corresponding Source for a non-source form of such a combination shall
// include the source code for the parts of OpenSSL used as well as that of
// the covered work.

//! Simple block-tiling GEMM for ARM
//!
//! This follows llama.cpp's approach for ARM: 16x16 block tiling with vec_dot.
//! Unlike the tiled GEMM (which is AVX-only), this approach:
//! - No complex unpack/repack overhead
//! - Directly processes packed quantized formats
//! - Uses NEON-optimized vec_dot per row
//! - 16x16 blocks for cache efficiency
//!
//! For GEMV (n=1), uses interleaved format to process 8 rows at once
//! For GEMM (n>1), falls back to simple row-by-row processing

use crate::cpu::SyncPtr;
use crate::quant::types::*;
use crate::quant::vec_dot::arm::k_quant::*;
use crate::quant::vec_dot::arm::k_quant_batch::{
    gemv_q4_k_interleaved, repack_q4_k_batch,
};
use std::sync::Barrier;

const INTERLEAVE_ROWS: usize = 8;

/// Q5_K × FP32 GEMM with pipelined quantization
/// This is the key optimization from llama.cpp: quantization happens on-demand during GEMM
/// weights: m × k (as BlockQ5K), inputs: n × k (FP32), outputs: m × n
/// Each thread quantizes only the inputs it needs, right before computing
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q5k_f32_pipelined(
    weights: &[BlockQ5K],
    inputs: &[&[f32]], // n inputs, each of size k
    dst: &mut [f32],
    k: usize,
    m: usize,
    n: usize,
    shared_q8k_buffer: &mut [BlockQ8K], // Shared buffer for quantized inputs
) {
    use crate::quant::kv_cache::quantize_row_q8_k_neon;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let nb = k / QK_K;
    let scheduler = crate::cpu::get_scheduler();
    let n_threads = scheduler.optimal_threads(m, k, n);

    // Shared quantization counter (work-stealing over inputs)
    let quant_counter = AtomicUsize::new(0);

    // Shared GEMM row counter (work-stealing over weight rows)
    let gemm_counter = AtomicUsize::new(n_threads);
    let row_chunk_size = if m >= n_threads * 16 { 16 } else { 64 };

    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
    let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ5K);
    let q8k_ptr = shared_q8k_buffer.as_mut_ptr() as usize; // Cast to usize for Send

    // Barrier to ensure all threads complete quantization before GEMM starts
    let barrier = std::sync::Arc::new(Barrier::new(n_threads));

    std::thread::scope(|s| {
        for ith in 0..n_threads {
            let dst_ptr = dst_ptr.clone();
            let weights_ptr = weights_ptr.clone();
            let quant_counter = &quant_counter;
            let gemm_counter = &gemm_counter;
            let barrier = barrier.clone();

            s.spawn(move || {
                let mut x_padded = vec![0.0f32; k];

                // Phase 1: Parallel quantization (work-stealing over inputs)
                loop {
                    let input_idx =
                        quant_counter.fetch_add(1, Ordering::Relaxed);
                    if input_idx >= n {
                        break;
                    }

                    let input = inputs[input_idx];
                    let x_q8k = std::slice::from_raw_parts_mut(
                        (q8k_ptr as *mut BlockQ8K).add(input_idx * nb),
                        nb,
                    );

                    if input.len() != k {
                        for i in input.len()..k {
                            x_padded[i] = 0.0;
                        }
                        x_padded[..input.len()].copy_from_slice(input);
                        quantize_row_q8_k_neon(&x_padded, x_q8k);
                    } else {
                        quantize_row_q8_k_neon(input, x_q8k);
                    }
                }

                // Wait for all threads to finish quantization
                barrier.wait();

                // Phase 2: Parallel GEMM (work-stealing over rows)
                // Each thread processes a chunk of rows, iterating through all columns
                // This keeps weights in L2 cache while processing all n inputs
                let mut row_idx = ith;
                while row_idx < m {
                    let row_start = row_idx * row_chunk_size;
                    let row_end = (row_start + row_chunk_size).min(m);

                    for row in row_start..row_end {
                        let w_row = std::slice::from_raw_parts(
                            weights_ptr.get().add(row * nb),
                            nb,
                        );

                        for j in 0..n {
                            let x_q8k = std::slice::from_raw_parts(
                                (q8k_ptr as *mut BlockQ8K).add(j * nb),
                                nb,
                            );
                            *dst_ptr.get().add(row * n + j) =
                                vec_dot_q5_k_q8_k_neon(k, w_row, x_q8k);
                        }
                    }

                    row_idx = gemm_counter.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
}

/// Q5_K × Q8_K GEMM using simple block-tiling
/// weights: m × k (as BlockQ5K), inputs: k × n (as BlockQ8K), outputs: m × n
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q5k_q8k_block_tiling(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
    n: usize,
) {
    let nb = k / QK_K;

    // Use adaptive scheduler for optimal thread count (matching llama.cpp)
    let scheduler = crate::cpu::get_scheduler();
    let n_threads = scheduler.optimal_threads(m, k, n);

    if n == 1 {
        // GEMV case: chunk_size = 64 (matching llama.cpp)
        let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
        let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ5K);
        let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8K);
        let weights_len = weights.len();
        let x_len = x.len();

        crate::cpu::parallel_chunks_sized(
            m,
            64, // GEMV chunk size
            n_threads,
            move |start, end| {
                for row in start..end {
                    let w_row = std::slice::from_raw_parts(
                        weights_ptr.get().add(row * nb),
                        nb,
                    );
                    let x_vec = std::slice::from_raw_parts(x_ptr.get(), x_len);
                    *dst_ptr.get().add(row) =
                        vec_dot_q5_k_q8_k_neon(k, w_row, x_vec);
                }
            },
        );
        return;
    }

    // GEMM case: chunk_size = 16 (matching llama.cpp)
    // Each thread caches its subset of weights and processes all n inputs
    // This keeps weights in L2 cache while processing all inputs
    if n_threads > 1 {
        // Parallel over rows for large m
        let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
        let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ5K);
        let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8K);

        crate::cpu::parallel_chunks_sized(
            m,
            16, // GEMM chunk size
            n_threads,
            move |start, end| {
                for row in start..end {
                    let w_row = std::slice::from_raw_parts(
                        weights_ptr.get().add(row * nb),
                        nb,
                    );
                    // Process all n inputs for this row
                    for j in 0..n {
                        let x_vec = std::slice::from_raw_parts(
                            x_ptr.get().add(j * nb),
                            nb,
                        );
                        *dst_ptr.get().add(row * n + j) =
                            vec_dot_q5_k_q8_k_neon(k, w_row, x_vec);
                    }
                }
            },
        );
    } else {
        // Sequential for small m
        for row in 0..m {
            let w_row = &weights[row * nb..(row + 1) * nb];
            for j in 0..n {
                let x_vec = &x[j * nb..(j + 1) * nb];
                dst[row * n + j] = vec_dot_q5_k_q8_k_neon(k, w_row, x_vec);
            }
        }
    }
}

/// Q4_K × FP32 GEMM with pipelined quantization
/// Each thread quantizes only the inputs it needs, right before computing
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q4k_f32_pipelined(
    weights: &[BlockQ4K],
    inputs: &[&[f32]], // n inputs, each of size k
    dst: &mut [f32],
    k: usize,
    m: usize,
    n: usize,
    shared_q8k_buffer: &mut [BlockQ8K], // Shared buffer for quantized inputs
) {
    use crate::quant::kv_cache::quantize_row_q8_k_neon;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let nb = k / QK_K;
    let scheduler = crate::cpu::get_scheduler();
    let n_threads = scheduler.optimal_threads(m, k, n);

    let quant_counter = AtomicUsize::new(0);
    let gemm_counter = AtomicUsize::new(n_threads);
    let row_chunk_size = if m >= n_threads * 16 { 16 } else { 64 };

    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
    let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ4K);
    let q8k_ptr = shared_q8k_buffer.as_mut_ptr() as usize;

    // Barrier to ensure all threads complete quantization before GEMM starts
    let barrier = std::sync::Arc::new(Barrier::new(n_threads));

    std::thread::scope(|s| {
        for ith in 0..n_threads {
            let dst_ptr = dst_ptr.clone();
            let weights_ptr = weights_ptr.clone();
            let quant_counter = &quant_counter;
            let gemm_counter = &gemm_counter;
            let barrier = barrier.clone();

            s.spawn(move || {
                let mut x_padded = vec![0.0f32; k];

                // Phase 1: Parallel quantization
                loop {
                    let input_idx =
                        quant_counter.fetch_add(1, Ordering::Relaxed);
                    if input_idx >= n {
                        break;
                    }

                    let input = inputs[input_idx];
                    let x_q8k = std::slice::from_raw_parts_mut(
                        (q8k_ptr as *mut BlockQ8K).add(input_idx * nb),
                        nb,
                    );

                    if input.len() != k {
                        for i in input.len()..k {
                            x_padded[i] = 0.0;
                        }
                        x_padded[..input.len()].copy_from_slice(input);
                        quantize_row_q8_k_neon(&x_padded, x_q8k);
                    } else {
                        quantize_row_q8_k_neon(input, x_q8k);
                    }
                }

                // Wait for all threads to finish quantization
                barrier.wait();

                // Phase 2: Parallel GEMM (work-stealing over rows)
                let mut row_idx = ith;
                while row_idx < m {
                    let row_start = row_idx * row_chunk_size;
                    let row_end = (row_start + row_chunk_size).min(m);

                    for row in row_start..row_end {
                        let w_row = std::slice::from_raw_parts(
                            weights_ptr.get().add(row * nb),
                            nb,
                        );

                        for j in 0..n {
                            let x_q8k = std::slice::from_raw_parts(
                                (q8k_ptr as *mut BlockQ8K).add(j * nb),
                                nb,
                            );
                            *dst_ptr.get().add(row * n + j) =
                                vec_dot_q4_k_q8_k_neon(k, w_row, x_q8k);
                        }
                    }

                    row_idx = gemm_counter.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
}

/// Q4_K × Q8_K GEMM using simple block-tiling
/// For GEMV (n=1), uses interleaved format to process 8 rows at once
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q4k_q8k_block_tiling(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
    n: usize,
) {
    let nb = k / QK_K;

    // Use adaptive scheduler for optimal thread count (matching llama.cpp)
    let scheduler = crate::cpu::get_scheduler();
    let n_threads = scheduler.optimal_threads(m, k, n);

    if n == 1 && m % INTERLEAVE_ROWS == 0 {
        // GEMV with interleaving: process 8 rows at once for better cache locality
        let weights_interleaved = repack_q4_k_batch(weights, m, nb);
        gemv_q4_k_interleaved(&weights_interleaved, x, dst, k, m);
        return;
    }

    if n == 1 {
        // GEMV without interleaving (m not divisible by 8): parallel vec_dot
        let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
        let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ4K);
        let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8K);

        crate::cpu::parallel_chunks_sized(
            m,
            64, // GEMV chunk size
            n_threads,
            move |start, end| {
                for row in start..end {
                    let w_row = std::slice::from_raw_parts(
                        weights_ptr.get().add(row * nb),
                        nb,
                    );
                    let x_vec = std::slice::from_raw_parts(x_ptr.get(), nb);
                    *dst_ptr.get().add(row) =
                        vec_dot_q4_k_q8_k_neon(k, w_row, x_vec);
                }
            },
        );
        return;
    }

    // GEMM case: parallelize over weight rows
    if n_threads > 1 {
        let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
        let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ4K);
        let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8K);

        crate::cpu::parallel_chunks_sized(
            m,
            16, // GEMM chunk size
            n_threads,
            move |start, end| {
                for row in start..end {
                    let w_row = std::slice::from_raw_parts(
                        weights_ptr.get().add(row * nb),
                        nb,
                    );
                    for j in 0..n {
                        let x_vec = std::slice::from_raw_parts(
                            x_ptr.get().add(j * nb),
                            nb,
                        );
                        *dst_ptr.get().add(row * n + j) =
                            vec_dot_q4_k_q8_k_neon(k, w_row, x_vec);
                    }
                }
            },
        );
    } else {
        for row in 0..m {
            let w_row = &weights[row * nb..(row + 1) * nb];
            for j in 0..n {
                let x_vec = &x[j * nb..(j + 1) * nb];
                dst[row * n + j] = vec_dot_q4_k_q8_k_neon(k, w_row, x_vec);
            }
        }
    }
}

/// Q6_K × FP32 GEMM with pipelined quantization
/// Each thread quantizes only the inputs it needs, right before computing
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q6k_f32_pipelined(
    weights: &[BlockQ6K],
    inputs: &[&[f32]], // n inputs, each of size k
    dst: &mut [f32],
    k: usize,
    m: usize,
    n: usize,
    shared_q8k_buffer: &mut [BlockQ8K], // Shared buffer for quantized inputs
) {
    use crate::quant::kv_cache::quantize_row_q8_k_neon;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let nb = k / QK_K;
    let scheduler = crate::cpu::get_scheduler();
    let n_threads = scheduler.optimal_threads(m, k, n);

    let quant_counter = AtomicUsize::new(0);
    let gemm_counter = AtomicUsize::new(n_threads);
    let row_chunk_size = if m >= n_threads * 16 { 16 } else { 64 };

    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
    let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ6K);
    let q8k_ptr = shared_q8k_buffer.as_mut_ptr() as usize;

    // Barrier to ensure all threads complete quantization before GEMM starts
    let barrier = std::sync::Arc::new(Barrier::new(n_threads));

    std::thread::scope(|s| {
        for ith in 0..n_threads {
            let dst_ptr = dst_ptr.clone();
            let weights_ptr = weights_ptr.clone();
            let quant_counter = &quant_counter;
            let gemm_counter = &gemm_counter;
            let barrier = barrier.clone();

            s.spawn(move || {
                let mut x_padded = vec![0.0f32; k];

                // Phase 1: Parallel quantization
                loop {
                    let input_idx =
                        quant_counter.fetch_add(1, Ordering::Relaxed);
                    if input_idx >= n {
                        break;
                    }

                    let input = inputs[input_idx];
                    let x_q8k = std::slice::from_raw_parts_mut(
                        (q8k_ptr as *mut BlockQ8K).add(input_idx * nb),
                        nb,
                    );

                    if input.len() != k {
                        for i in input.len()..k {
                            x_padded[i] = 0.0;
                        }
                        x_padded[..input.len()].copy_from_slice(input);
                        quantize_row_q8_k_neon(&x_padded, x_q8k);
                    } else {
                        quantize_row_q8_k_neon(input, x_q8k);
                    }
                }

                // Wait for all threads to finish quantization
                barrier.wait();

                // Phase 2: Parallel GEMM (work-stealing over rows)
                let mut row_idx = ith;
                while row_idx < m {
                    let row_start = row_idx * row_chunk_size;
                    let row_end = (row_start + row_chunk_size).min(m);

                    for row in row_start..row_end {
                        let w_row = std::slice::from_raw_parts(
                            weights_ptr.get().add(row * nb),
                            nb,
                        );

                        for j in 0..n {
                            let x_q8k = std::slice::from_raw_parts(
                                (q8k_ptr as *mut BlockQ8K).add(j * nb),
                                nb,
                            );
                            *dst_ptr.get().add(row * n + j) =
                                vec_dot_q6_k_q8_k_neon(k, w_row, x_q8k);
                        }
                    }

                    row_idx = gemm_counter.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
}

/// Q6_K × Q8_K GEMM using simple block-tiling
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q6k_q8k_block_tiling(
    weights: &[BlockQ6K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
    n: usize,
) {
    let nb = k / QK_K;

    // Use adaptive scheduler for optimal thread count (matching llama.cpp)
    let scheduler = crate::cpu::get_scheduler();
    let n_threads = scheduler.optimal_threads(m, k, n);

    if n == 1 {
        // GEMV: parallel vec_dot
        let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
        let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ6K);
        let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8K);

        crate::cpu::parallel_chunks_sized(
            m,
            64, // GEMV chunk size
            n_threads,
            move |start, end| {
                for row in start..end {
                    let w_row = std::slice::from_raw_parts(
                        weights_ptr.get().add(row * nb),
                        nb,
                    );
                    let x_vec = std::slice::from_raw_parts(x_ptr.get(), nb);
                    *dst_ptr.get().add(row) =
                        vec_dot_q6_k_q8_k_neon(k, w_row, x_vec);
                }
            },
        );
        return;
    }

    // GEMM case: parallelize over weight rows
    if n_threads > 1 {
        let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
        let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ6K);
        let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8K);

        crate::cpu::parallel_chunks_sized(
            m,
            16, // GEMM chunk size
            n_threads,
            move |start, end| {
                for row in start..end {
                    let w_row = std::slice::from_raw_parts(
                        weights_ptr.get().add(row * nb),
                        nb,
                    );
                    for j in 0..n {
                        let x_vec = std::slice::from_raw_parts(
                            x_ptr.get().add(j * nb),
                            nb,
                        );
                        *dst_ptr.get().add(row * n + j) =
                            vec_dot_q6_k_q8_k_neon(k, w_row, x_vec);
                    }
                }
            },
        );
    } else {
        for row in 0..m {
            let w_row = &weights[row * nb..(row + 1) * nb];
            for j in 0..n {
                let x_vec = &x[j * nb..(j + 1) * nb];
                dst[row * n + j] = vec_dot_q6_k_q8_k_neon(k, w_row, x_vec);
            }
        }
    }
}
