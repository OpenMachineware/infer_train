// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

//! Q8_0 GEMM kernel implementation.

use super::{GemmKernel, Q8_0};
use crate::cpu::{SyncPtr, parallel_chunks};
use crate::quant::types::BlockQ8_0;
use half::f16;

const QK8_0: usize = 32; // Block size for Q8_0

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

impl GemmKernel for Q8_0 {
    type Weight = BlockQ8_0;

    #[inline]
    fn gemm(
        c: &mut [f32],
        a: &[BlockQ8_0],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        threads: usize,
    ) {
        let nb = k / QK8_0;
        debug_assert_eq!(a.len(), m * nb);
        debug_assert_eq!(b.len(), n * k);
        debug_assert_eq!(c.len(), m * n);

        gemm_q8_0_simd(c, a, b, m, n, k, nb, threads);
    }
}

fn gemm_q8_0_simd(
    c: &mut [f32],
    a: &[BlockQ8_0],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
    threads: usize,
) {
    if threads <= 1 || m < 64 {
        unsafe {
            gemm_q8_0_single(
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
            gemm_q8_0_parallel(c, a, b, m, n, k, nb, threads);
        }
    }
}

/// Q8_0: Simple 8-bit quantization with scale
/// Each block: scale (f16) + 32 int8 values
#[target_feature(enable = "neon")]
unsafe fn gemm_q8_0_single(
    c: *mut f32,
    a: *const BlockQ8_0,
    b: *const f32,
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
) {
    for i in 0..m {
        let row_a = a.add(i * nb);
        let row_c = c.add(i * n);

        for j in 0..n {
            let col_b = b.add(j * k);
            let mut sum = 0.0f32;

            for block_idx in 0..nb {
                let block = &*row_a.add(block_idx);
                let scale = f16_to_f32(block.d);
                let block_k = block_idx * QK8_0;

                // Simple scalar dot product for now
                // SIMD can be added later with proper widening
                let mut dot = 0.0f32;
                for q_idx in 0..QK8_0 {
                    dot += block.qs[q_idx] as f32 * *col_b.add(block_k + q_idx);
                }
                sum += dot * scale;
            }

            *row_c.add(j) = sum;
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn gemm_q8_0_parallel(
    c: &mut [f32],
    a: &[BlockQ8_0],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
    threads: usize,
) {
    let c_ptr = SyncPtr::new(c.as_mut_ptr());
    let a_ptr = SyncPtr::new(a.as_ptr() as *mut BlockQ8_0);
    let b_ptr = SyncPtr::new(b.as_ptr() as *mut f32);

    parallel_chunks(m, threads, move |start, end| {
        for i in start..end {
            let row_a = a_ptr.get().add(i * nb);
            let row_c = c_ptr.get().add(i * n);

            for j in 0..n {
                let col_b = b_ptr.get().add(j * k);
                let mut sum = 0.0f32;

                for block_idx in 0..nb {
                    let block = &*row_a.add(block_idx);
                    let scale = f16_to_f32(block.d);
                    let block_k = block_idx * QK8_0;

                    for q_idx in 0..QK8_0 {
                        sum += block.qs[q_idx] as f32
                            * scale
                            * *col_b.add(block_k + q_idx);
                    }
                }

                *row_c.add(j) = sum;
            }
        }
    });
}
