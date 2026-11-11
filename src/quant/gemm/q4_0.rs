// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

//! Q4_0 GEMM kernel implementation.
//!
//! For large batches (n >= 32): dequantize all weights and use BLAS
//! For small batches (n < 32): use on-the-fly dequantization with SIMD

use super::{GemmKernel, Q4_0};
use crate::cpu::{SyncPtr, parallel_chunks};
use crate::quant::types::BlockQ4_0;
use half::f16;

const QK4_0: usize = 32; // Block size for Q4_0

/// Convert FP16 bits to f32
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

impl GemmKernel for Q4_0 {
    type Weight = BlockQ4_0;

    #[inline]
    fn gemm(
        c: &mut [f32],
        a: &[BlockQ4_0],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        threads: usize,
    ) {
        let nb = k / QK4_0;
        debug_assert_eq!(a.len(), m * nb);
        debug_assert_eq!(b.len(), n * k);
        debug_assert_eq!(c.len(), m * n);

        // For large batches, use BLAS path
        // TODO: Integrate BLAS for n >= 32
        gemm_q4_0_simd(c, a, b, m, n, k, nb, threads);
    }
}

/// SIMD GEMM for Q4_0 with on-the-fly dequantization
fn gemm_q4_0_simd(
    c: &mut [f32],
    a: &[BlockQ4_0],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
    threads: usize,
) {
    if threads <= 1 || m < 64 {
        unsafe {
            gemm_q4_0_single(
                c.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                m,
                n,
                k,
                nb,
            );
        }
    } else {
        unsafe {
            gemm_q4_0_parallel(c, a, b, m, n, k, nb, threads);
        }
    }
}

/// Single-threaded Q4_0 GEMM
#[target_feature(enable = "neon")]
unsafe fn gemm_q4_0_single(
    c: *mut f32,
    a: *const BlockQ4_0,
    b: *const f32,
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
) {
    // Process each row
    for i in 0..m {
        let row_a = a.add(i * nb);
        let row_c = c.add(i * n);

        // Process each column
        for j in 0..n {
            let col_b = b.add(j * k);
            let mut sum = 0.0f32;

            // Accumulate over blocks
            for block_idx in 0..nb {
                let block = &*row_a.add(block_idx);
                let scale = f16_to_f32(block.d);
                let block_k = block_idx * QK4_0;

                // Dequantize and compute dot product
                for q_idx in 0..QK4_0 / 2 {
                    let packed = block.qs[q_idx];
                    let v0 = ((packed & 0x0F) as i8 - 8) as f32 * scale;
                    let v1 = ((packed >> 4) as i8 - 8) as f32 * scale;

                    sum += v0 * *col_b.add(block_k + q_idx * 2);
                    sum += v1 * *col_b.add(block_k + q_idx * 2 + 1);
                }
            }

            *row_c.add(j) = sum;
        }
    }
}

/// Parallel Q4_0 GEMM
#[target_feature(enable = "neon")]
unsafe fn gemm_q4_0_parallel(
    c: &mut [f32],
    a: &[BlockQ4_0],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
    threads: usize,
) {
    let c_ptr = SyncPtr::new(c.as_mut_ptr());
    let a_ptr = SyncPtr::new(a.as_ptr() as *mut BlockQ4_0);
    let b_ptr = SyncPtr::new(b.as_ptr() as *mut f32);

    parallel_chunks(m, threads, move |start, end| {
        for i in start..end {
            gemm_q4_0_row(
                c_ptr.get().add(i * n),
                a_ptr.get().add(i * nb),
                b_ptr.get(),
                n,
                k,
                nb,
            );
        }
    });
}

/// Process one row of Q4_0 GEMM
#[target_feature(enable = "neon")]
#[inline]
unsafe fn gemm_q4_0_row(
    c: *mut f32,
    a: *const BlockQ4_0,
    b: *const f32,
    n: usize,
    k: usize,
    nb: usize,
) {
    for j in 0..n {
        let col_b = b.add(j * k);
        let mut sum = 0.0f32;

        // Process blocks
        for block_idx in 0..nb {
            let block = &*a.add(block_idx);
            let scale = f16_to_f32(block.d);
            let block_k = block_idx * QK4_0;

            for q_idx in 0..QK4_0 / 2 {
                let packed = block.qs[q_idx];
                let v0 = ((packed & 0x0F) as i8 - 8) as f32 * scale;
                let v1 = ((packed >> 4) as i8 - 8) as f32 * scale;

                sum += v0 * *col_b.add(block_k + q_idx * 2);
                sum += v1 * *col_b.add(block_k + q_idx * 2 + 1);
            }
        }

        *c.add(j) = sum;
    }
}
