// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

//! FP32 GEMM kernel implementation.

use super::{F32, GemmKernel};
use crate::cpu::{SyncPtr, parallel_chunks};
use std::arch::aarch64::*;

impl GemmKernel for F32 {
    type Weight = f32;

    #[inline]
    fn gemm(
        c: &mut [f32],
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        threads: usize,
    ) {
        debug_assert_eq!(a.len(), m * k);
        debug_assert_eq!(b.len(), n * k);
        debug_assert_eq!(c.len(), m * n);

        // Use BLAS for large batch sizes (prefill path)
        #[cfg(target_os = "macos")]
        {
            if n >= super::BLAS_THRESHOLD {
                unsafe {
                    super::blas::gemm_blas_f32(
                        c.as_mut_ptr(),
                        a.as_ptr(),
                        b.as_ptr(),
                        m,
                        n,
                        k,
                    );
                }
                return;
            }
        }

        // For small batches, use SIMD path
        if threads <= 1 || m < 64 {
            unsafe {
                sgemm_single(c.as_mut_ptr(), a.as_ptr(), b.as_ptr(), m, k, n);
            }
        } else {
            unsafe {
                sgemm_parallel(c, a, b, m, k, n, threads);
            }
        }
    }
}

/// Single-threaded SIMD GEMM
#[target_feature(enable = "neon")]
unsafe fn sgemm_single(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    m: usize,
    k: usize,
    n: usize,
) {
    // Process in 4-row blocks
    let mut ii = 0;
    while ii + 4 <= m {
        process_block_4x16(c.add(ii * n), a.add(ii * k), b, k, n);
        ii += 4;
    }

    // Handle remaining rows
    while ii < m {
        process_row(c.add(ii * n), a.add(ii * k), b, k, n);
        ii += 1;
    }
}

/// Process 4 rows with 16-column blocking
///
/// Matrix layout:
/// - A: [m, k] row-major
/// - B: [n, k] row-major
/// - C: [m, n] row-major
///
/// Computes C[i, j] = sum_l A[i, l] * B[j, l]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn process_block_4x16(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    k: usize,
    n: usize,
) {
    const RM: usize = 4;
    const RN: usize = 4; // 4 * EPR = 16 columns
    const EPR: usize = 4;

    // Process 16 columns at a time
    let mut jj = 0;
    while jj + RN * EPR <= n {
        // Accumulators for 4 rows x 16 columns
        let mut acc: [[float32x4_t; RN]; RM] = std::mem::zeroed();

        // Load initial C values
        for i in 0..RM {
            for r in 0..RN {
                acc[i][r] = vld1q_f32(c.add(i * n + jj + r * EPR));
            }
        }

        // K loop
        for kk in 0..k {
            // Load A column (4 elements) and broadcast
            for i in 0..RM {
                let av = vdupq_n_f32(*a.add(i * k + kk));

                // Load B values for columns jj to jj+15
                // B[j, kk] is at b[j * k + kk]
                // For 16 columns j=jj to jj+15, we load B[j, kk]
                for r in 0..RN {
                    let bv = vld1q_f32(b.add((jj + r * EPR) * k + kk));
                    acc[i][r] = vfmaq_f32(acc[i][r], bv, av);
                }
            }
        }

        // Store results
        for i in 0..RM {
            for r in 0..RN {
                vst1q_f32(c.add(i * n + jj + r * EPR), acc[i][r]);
            }
        }

        jj += RN * EPR;
    }

    // Handle remaining columns
    while jj < n {
        for i in 0..RM {
            let mut sum = *c.add(i * n + jj);
            for kk in 0..k {
                sum += *a.add(i * k + kk) * *b.add(jj * k + kk);
            }
            *c.add(i * n + jj) = sum;
        }
        jj += 1;
    }
}

/// Process single row
#[target_feature(enable = "neon")]
#[inline]
unsafe fn process_row(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    k: usize,
    n: usize,
) {
    // Process 4 columns at a time
    let mut jj = 0;
    while jj + 4 <= n {
        let mut acc = vld1q_f32(c.add(jj));

        for kk in 0..k {
            let av = vdupq_n_f32(*a.add(kk));
            // B[j, kk] is at b[j * k + kk]
            let bv = vld1q_f32(b.add(jj * k + kk));
            acc = vfmaq_f32(acc, bv, av);
        }

        vst1q_f32(c.add(jj), acc);
        jj += 4;
    }

    // Handle remaining columns
    while jj < n {
        let mut sum = *c.add(jj);
        for kk in 0..k {
            sum += *a.add(kk) * *b.add(jj * k + kk);
        }
        *c.add(jj) = sum;
        jj += 1;
    }
}

/// Parallel GEMM using work-stealing
#[target_feature(enable = "neon")]
unsafe fn sgemm_parallel(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    threads: usize,
) {
    let c_ptr = SyncPtr::new(c.as_mut_ptr());
    let a_ptr = SyncPtr::new(a.as_ptr() as *mut f32);
    let b_ptr = SyncPtr::new(b.as_ptr() as *mut f32);

    parallel_chunks(m, threads, move |start, end| {
        let chunk_m = end - start;
        sgemm_single(
            c_ptr.get().add(start * n),
            a_ptr.get().add(start * k),
            b_ptr.get(),
            chunk_m,
            k,
            n,
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test FP32 GEMM: C = A * B^T
    /// A: [m, k], B: [n, k], C: [m, n]
    /// C[i, j] = sum_l A[i, l] * B[j, l]
    #[test]
    fn test_fp32_gemm() {
        let m = 16;
        let n = 8;
        let k = 32;

        let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i * 2) as f32).collect();
        let mut c = vec![0.0f32; m * n];

        F32::gemm(&mut c, &a, &b, m, n, k, 1);

        // Verify first element: C[0, 0] = A[0, :] · B[0, :]
        let mut expected = 0.0f32;
        for l in 0..k {
            expected += a[l] * b[l]; // A[0, l] * B[0, l]
        }
        assert!(
            (c[0] - expected).abs() < 1e-3,
            "C[0]={}, expected={}",
            c[0],
            expected
        );

        // Verify second column: C[0, 1] = A[0, :] · B[1, :]
        let mut expected1 = 0.0f32;
        for l in 0..k {
            expected1 += a[l] * b[k + l]; // A[0, l] * B[1, l]
        }
        assert!(
            (c[1] - expected1).abs() < 1e-3,
            "C[1]={}, expected={}",
            c[1],
            expected1
        );
    }
}
