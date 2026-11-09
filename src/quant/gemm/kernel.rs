// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

//! SIMD-optimized FP32 GEMM kernel.
//!
//! This implements the core matrix multiplication microkernel
//! used by all quantized GEMM variants after dequantization.
//!
//! # Algorithm
//!
//! Uses 4x4 register blocking for ARM NEON:
//! - Process 4 rows of A at a time
//! - Process 4 columns of B at a time
//! - Accumulate in SIMD registers to minimize memory traffic

use std::arch::aarch64::*;

/// Register blocking parameters for ARM NEON (32 registers)
const RM: usize = 4; // Rows per microkernel
const RN: usize = 4; // Columns per microkernel (in terms of float32x4_t)
const EPR: usize = 4; // Elements per NEON register

/// SIMD GEMM microkernel: C[RM x RN*EPR] += A[RM x K] * B[K x RN*EPR]
///
/// Uses 4x16 register blocking:
/// - 4 accumulator registers per row (16 total)
/// - 4 B vector registers
/// - 1 broadcast register for A
/// Total: 21 registers (within 32 limit)
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn sgemm_ukernel(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    k: usize,
    n: usize, // ldc
) {
    // Accumulator registers [RM][RN]
    // Each acc[i][r] holds 4 floats for column r*4 to (r+1)*4
    let mut acc: [[float32x4_t; RN]; RM] = std::mem::zeroed();

    // Load initial C values
    for i in 0..RM {
        for r in 0..RN {
            acc[i][r] = vld1q_f32(c.add(i * n + r * EPR));
        }
    }

    // Main K loop
    let k4 = k - k % 4;
    for kk in (0..k4).step_by(4) {
        // Load B columns (4 vectors of 4 floats each = 16 columns)
        let mut bv: [float32x4_t; RN] = std::mem::zeroed();
        for r in 0..RN {
            bv[r] = vld1q_f32(b.add(kk * n + r * EPR));
        }

        // Load A values and broadcast to each row
        // Process 4 K elements at once for better ILP
        for i in 0..RM {
            let a0 = vld1q_f32(a.add(i * k + kk));

            // FMA with each B vector
            for r in 0..RN {
                // Horizontally sum A*B for this row
                let p = vdupq_n_f32(vaddvq_f32(vmulq_f32(a0, bv[r])));
                acc[i][r] = vfmaq_f32(acc[i][r], bv[r], p);
            }
        }
    }

    // Handle remaining K elements (scalar fallback)
    for kk in k4..k {
        let mut bv: [float32x4_t; RN] = std::mem::zeroed();
        for r in 0..RN {
            bv[r] = vld1q_f32(b.add(kk * n + r * EPR));
        }

        for i in 0..RM {
            let av = vdupq_n_f32(*a.add(i * k + kk));
            for r in 0..RN {
                acc[i][r] = vfmaq_f32(acc[i][r], bv[r], av);
            }
        }
    }

    // Store results
    for i in 0..RM {
        for r in 0..RN {
            vst1q_f32(c.add(i * n + r * EPR), acc[i][r]);
        }
    }
}

/// Main SIMD GEMM: C[m,n] += A[m,k]^T * B[n,k]
///
/// - A: weights [m, k] in row-major
/// - B: inputs [n, k] in row-major
/// - C: output [m, n] in row-major
///
/// Note: This computes A^T * B, which is equivalent to
/// treating A's columns as rows.
#[target_feature(enable = "neon")]
pub unsafe fn sgemm(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    m: usize,
    k: usize,
    n: usize,
) {
    // Process in RM-row blocks
    let mut ii = 0;
    while ii + RM <= m {
        // Process in 16-column blocks (RN * EPR)
        let mut jj = 0;
        while jj + RN * EPR <= n {
            sgemm_ukernel(c.add(ii * n + jj), a.add(ii * k), b.add(jj), k, n);
            jj += RN * EPR;
        }

        // Handle remaining columns with scalar fallback
        while jj < n {
            for i in 0..RM {
                let mut sum = *c.add((ii + i) * n + jj);
                for kk in 0..k {
                    sum += *a.add((ii + i) * k + kk) * *b.add(jj * k + kk);
                }
                *c.add((ii + i) * n + jj) = sum;
            }
            jj += 1;
        }

        ii += RM;
    }

    // Handle remaining rows
    while ii < m {
        let mut jj = 0;
        while jj < n {
            let mut sum = *c.add(ii * n + jj);
            for kk in 0..k {
                sum += *a.add(ii * k + kk) * *b.add(jj * k + kk);
            }
            *c.add(ii * n + jj) = sum;
            jj += 1;
        }
        ii += 1;
    }
}

/// Parallel GEMM using work-stealing scheduler
#[target_feature(enable = "neon")]
pub unsafe fn sgemm_parallel(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) {
    use crate::cpu::{SyncPtr, parallel_chunks};

    let c_ptr = SyncPtr::new(c.as_mut_ptr());
    let a_ptr = SyncPtr::new(a.as_ptr() as *mut f32);
    let b_ptr = SyncPtr::new(b.as_ptr() as *mut f32);

    // Parallelize over rows of C
    parallel_chunks(m, threads, move |start, end| {
        let chunk_m = end - start;
        sgemm(
            c_ptr.get().add(start * n),
            a_ptr.get().add(start * k),
            b_ptr.get(),
            chunk_m,
            k,
            n,
        );
    });
}
