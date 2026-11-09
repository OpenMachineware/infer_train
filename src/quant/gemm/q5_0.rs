// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

//! Q5_0 GEMM kernel implementation.

use super::{GemmKernel, Q5_0};
use crate::cpu::{SyncPtr, parallel_chunks};
use crate::quant::types::BlockQ5_0;
use half::f16;
use std::arch::aarch64::*;

const QK5_0: usize = 32; // Block size for Q5_0

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

impl GemmKernel for Q5_0 {
    type Weight = BlockQ5_0;

    #[inline]
    fn gemm(
        c: &mut [f32],
        a: &[BlockQ5_0],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        threads: usize,
    ) {
        let nb = k / QK5_0;
        debug_assert_eq!(a.len(), m * nb);
        debug_assert_eq!(b.len(), n * k);
        debug_assert_eq!(c.len(), m * n);

        gemm_q5_0_simd(c, a, b, m, n, k, nb, threads);
    }
}

/// SIMD GEMM for Q5_0
fn gemm_q5_0_simd(
    c: &mut [f32],
    a: &[BlockQ5_0],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
    threads: usize,
) {
    if threads <= 1 || m < 64 {
        unsafe {
            gemm_q5_0_single(
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
            gemm_q5_0_parallel(c, a, b, m, n, k, nb, threads);
        }
    }
}

/// Q5_0 dequantization: 5-bit packed in qs[16] + qh[4]
/// Each block has 32 values: low 4 bits in qs, high 1 bit in qh
#[inline]
fn dequantize_q5_0(block: &BlockQ5_0, idx: usize) -> f32 {
    let scale = f16_to_f32(block.d);
    let l = idx / 2;
    let is_high = idx % 2 == 1;

    let low4 = if is_high { block.qs[l] >> 4 } else { block.qs[l] & 0x0F };

    // High bit from qh
    let qh_idx = idx;
    let high_bit = ((block.qh[qh_idx / 8] >> (qh_idx % 8)) & 1) as u8;

    let value = (low4 | (high_bit << 4)) as i32 - 16;
    value as f32 * scale
}

#[target_feature(enable = "neon")]
unsafe fn gemm_q5_0_single(
    c: *mut f32,
    a: *const BlockQ5_0,
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
                let block_k = block_idx * QK5_0;

                for q_idx in 0..QK5_0 {
                    let v = dequantize_q5_0(block, q_idx);
                    sum += v * *col_b.add(block_k + q_idx);
                }
            }

            *row_c.add(j) = sum;
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn gemm_q5_0_parallel(
    c: &mut [f32],
    a: &[BlockQ5_0],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
    threads: usize,
) {
    let c_ptr = SyncPtr::new(c.as_mut_ptr());
    let a_ptr = SyncPtr::new(a.as_ptr() as *mut BlockQ5_0);
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
                    let block_k = block_idx * QK5_0;

                    for q_idx in 0..QK5_0 {
                        sum += dequantize_q5_0(block, q_idx)
                            * *col_b.add(block_k + q_idx);
                    }
                }

                *row_c.add(j) = sum;
            }
        }
    });
}
