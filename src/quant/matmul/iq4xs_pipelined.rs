// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

// Pipelined IQ4_XS GEMM implementation

use crate::cpu::SyncPtr;
use crate::quant::kv_cache::quantize_row_q8_k_neon;
use crate::quant::types::{BlockIQ4XS, BlockQ8K, QK_K};
use crate::quant::vec_dot::arm::iq::vec_dot_iq4_xs_q8_k_neon;
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Pipelined IQ4_XS × Q8_K GEMM
/// Phase 1: Parallel quantization of all inputs
/// Phase 2: Parallel GEMM over output rows
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq4xs_f32_pipelined(
    weights: &[BlockIQ4XS],
    inputs: &[&[f32]], // n inputs, each of size k
    dst: &mut [f32],
    k: usize,
    m: usize,
    n: usize,
    shared_q8k_buffer: &mut [BlockQ8K], // Shared buffer for quantized inputs
) {
    let nb = k / QK_K;
    let scheduler = crate::cpu::get_scheduler();
    let n_threads = scheduler.optimal_threads(m, k, n);

    // Shared quantization counter (work-stealing over inputs)
    let quant_counter = AtomicUsize::new(0);

    // Shared GEMM row counter (work-stealing over rows)
    let gemm_counter = AtomicUsize::new(n_threads);
    let row_chunk_size = if m >= n_threads * 16 { 16 } else { 64 };

    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
    let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockIQ4XS);
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
                            // Col-major output: dst[j * m + row]
                            *dst_ptr.get().add(j * m + row) =
                                vec_dot_iq4_xs_q8_k_neon(k, w_row, x_q8k);
                        }
                    }

                    row_idx = gemm_counter.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
}
