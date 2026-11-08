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
    BlockQ4Kx8, gemv_q4_k_interleaved, repack_q4_k_batch,
};
use std::arch::aarch64::*;

const BLOCK_SIZE: usize = 16;
const INTERLEAVE_ROWS: usize = 8;

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

    if n == 1 {
        // GEMV case: use simple parallel vec_dot
        let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
        let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ5K);
        let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8K);
        let weights_len = weights.len();
        let x_len = x.len();

        crate::cpu::parallel_chunks(
            m,
            crate::cpu::default_n_threads(),
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

    // GEMM case: 16x16 block tiling
    // Process blocks of 16 output rows at a time
    for i_block_start in (0..m).step_by(BLOCK_SIZE) {
        let i_block_end = (i_block_start + BLOCK_SIZE).min(m);
        let n_rows = i_block_end - i_block_start;

        // Process blocks of 16 input columns at a time
        for j_block_start in (0..n).step_by(BLOCK_SIZE) {
            let j_block_end = (j_block_start + BLOCK_SIZE).min(n);
            let n_cols = j_block_end - j_block_start;

            // For each row in this block
            for i in 0..n_rows {
                let row_global = i_block_start + i;
                let w_row = &weights[row_global * nb..(row_global + 1) * nb];

                // For each column in this block
                for j in 0..n_cols {
                    let col_global = j_block_start + j;
                    let x_vec = &x[col_global * nb..(col_global + 1) * nb];

                    // Call vec_dot directly - no unpack overhead
                    dst[row_global * n + col_global] =
                        vec_dot_q5_k_q8_k_neon(k, w_row, x_vec);
                }
            }
        }
    }
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

        crate::cpu::parallel_chunks(
            m,
            crate::cpu::default_n_threads(),
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

    // GEMM: 16x16 block tiling
    for i_block_start in (0..m).step_by(BLOCK_SIZE) {
        let i_block_end = (i_block_start + BLOCK_SIZE).min(m);
        let n_rows = i_block_end - i_block_start;

        for j_block_start in (0..n).step_by(BLOCK_SIZE) {
            let j_block_end = (j_block_start + BLOCK_SIZE).min(n);
            let n_cols = j_block_end - j_block_start;

            for i in 0..n_rows {
                let row_global = i_block_start + i;
                let w_row = &weights[row_global * nb..(row_global + 1) * nb];

                for j in 0..n_cols {
                    let col_global = j_block_start + j;
                    let x_vec = &x[col_global * nb..(col_global + 1) * nb];

                    dst[row_global * n + col_global] =
                        vec_dot_q4_k_q8_k_neon(k, w_row, x_vec);
                }
            }
        }
    }
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

    if n == 1 {
        // GEMV: parallel vec_dot
        let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
        let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ6K);
        let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8K);

        crate::cpu::parallel_chunks(
            m,
            crate::cpu::default_n_threads(),
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

    // GEMM: 16x16 block tiling
    for i_block_start in (0..m).step_by(BLOCK_SIZE) {
        let i_block_end = (i_block_start + BLOCK_SIZE).min(m);
        let n_rows = i_block_end - i_block_start;

        for j_block_start in (0..n).step_by(BLOCK_SIZE) {
            let j_block_end = (j_block_start + BLOCK_SIZE).min(n);
            let n_cols = j_block_end - j_block_start;

            for i in 0..n_rows {
                let row_global = i_block_start + i;
                let w_row = &weights[row_global * nb..(row_global + 1) * nb];

                for j in 0..n_cols {
                    let col_global = j_block_start + j;
                    let x_vec = &x[col_global * nb..(col_global + 1) * nb];

                    dst[row_global * n + col_global] =
                        vec_dot_q6_k_q8_k_neon(k, w_row, x_vec);
                }
            }
        }
    }
}
