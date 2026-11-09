// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

//! IQ4_NL GEMM kernel implementation.

use super::{GemmKernel, IQ4_NL};
use crate::cpu::{SyncPtr, parallel_chunks};
use crate::quant::types::BlockIQ4NL;
use half::f16;
use std::arch::aarch64::*;

const QK4_NL: usize = 32; // Block size for IQ4_NL

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    f16::from_bits(bits).to_f32()
}

impl GemmKernel for IQ4_NL {
    type Weight = BlockIQ4NL;

    #[inline]
    fn gemm(
        c: &mut [f32],
        a: &[BlockIQ4NL],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        threads: usize,
    ) {
        let nb = k / QK4_NL;
        debug_assert_eq!(a.len(), m * nb);
        debug_assert_eq!(b.len(), n * k);
        debug_assert_eq!(c.len(), m * n);

        gemm_iq4_nl_simd(c, a, b, m, n, k, nb, threads);
    }
}

fn gemm_iq4_nl_simd(
    c: &mut [f32],
    a: &[BlockIQ4NL],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
    threads: usize,
) {
    if threads <= 1 || m < 64 {
        unsafe {
            gemm_iq4_nl_single(
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
            gemm_iq4_nl_parallel(c, a, b, m, n, k, nb, threads);
        }
    }
}

/// IQ4_NL: 4-bit quantization with lookup table
/// Values are indexed via 4-bit indices into a quantization grid
#[target_feature(enable = "neon")]
unsafe fn gemm_iq4_nl_single(
    c: *mut f32,
    a: *const BlockIQ4NL,
    b: *const f32,
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
) {
    // IQ4_NL grid (from llama.cpp)
    const KSCALE_IQ4_NL: usize = 8;
    const KVALUES_IQ4_NL: [i8; 16] =
        [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1];

    for i in 0..m {
        let row_a = a.add(i * nb);
        let row_c = c.add(i * n);

        for j in 0..n {
            let col_b = b.add(j * k);
            let mut sum = 0.0f32;

            for block_idx in 0..nb {
                let block = &*row_a.add(block_idx);
                let scale = f16_to_f32(block.d);
                let block_k = block_idx * QK4_NL;

                for q_idx in 0..QK4_NL / 2 {
                    let packed = block.qs[q_idx];

                    // Two 4-bit values
                    let v0 =
                        KVALUES_IQ4_NL[(packed & 0x0F) as usize] as f32 * scale;
                    let v1 =
                        KVALUES_IQ4_NL[(packed >> 4) as usize] as f32 * scale;

                    sum += v0 * *col_b.add(block_k + q_idx * 2);
                    sum += v1 * *col_b.add(block_k + q_idx * 2 + 1);
                }
            }

            *row_c.add(j) = sum;
        }
    }
}

#[target_feature(enable = "neon")]
unsafe fn gemm_iq4_nl_parallel(
    c: &mut [f32],
    a: &[BlockIQ4NL],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    nb: usize,
    threads: usize,
) {
    const KVALUES_IQ4_NL: [i8; 16] =
        [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1];

    let c_ptr = SyncPtr::new(c.as_mut_ptr());
    let a_ptr = SyncPtr::new(a.as_ptr() as *mut BlockIQ4NL);
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
                    let block_k = block_idx * QK4_NL;

                    for q_idx in 0..QK4_NL / 2 {
                        let packed = block.qs[q_idx];
                        let v0 = KVALUES_IQ4_NL[(packed & 0x0F) as usize]
                            as f32
                            * scale;
                        let v1 = KVALUES_IQ4_NL[(packed >> 4) as usize] as f32
                            * scale;

                        sum += v0 * *col_b.add(block_k + q_idx * 2);
                        sum += v1 * *col_b.add(block_k + q_idx * 2 + 1);
                    }
                }

                *row_c.add(j) = sum;
            }
        }
    });
}
