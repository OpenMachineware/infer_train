// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

//! SIMD-optimized GEMM kernel ported from llama.cpp
//! Computes C[M x N] += A[M x K] * B[K x N]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
const GEMM_RM: usize = 4; // Register blocking for M dimension
#[cfg(target_arch = "aarch64")]
const GEMM_RN: usize = 4; // Register blocking for N dimension
#[cfg(target_arch = "aarch64")]
const GGML_F32_EPR: usize = 4; // Elements per NEON register

/// SIMD GEMM microkernel: C[RM x RN*KN] += A[RM x K] * B[K x RN*KN]
/// Using 4x4 register blocking for optimal cache efficiency
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn simd_gemm_ukernel<const RM: usize, const RN: usize>(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    k: usize,
    n: usize,
) {
    const KN: usize = GGML_F32_EPR;

    // Accumulator registers [RM][RN]
    let mut acc: [[float32x4_t; RN]; RM] = std::mem::zeroed();

    // Load initial C values
    for i in 0..RM {
        for r in 0..RN {
            acc[i][r] = vld1q_f32(c.add(i * n + r * KN));
        }
    }

    // Main K loop
    for kk in 0..k {
        // Load B columns
        let mut bv: [float32x4_t; RN] = std::mem::zeroed();
        for r in 0..RN {
            bv[r] = vld1q_f32(b.add(kk * n + r * KN));
        }

        // Broadcast A and FMA with B
        for i in 0..RM {
            let p = vdupq_n_f32(*a.add(i * k + kk));
            for r in 0..RN {
                acc[i][r] = vfmaq_f32(acc[i][r], bv[r], p);
            }
        }
    }

    // Store results
    for i in 0..RM {
        for r in 0..RN {
            vst1q_f32(c.add(i * n + r * KN), acc[i][r]);
        }
    }
}

/// Handle tail columns (N % (RN*KN) != 0)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn simd_gemm_ukernel_tail<const RM: usize>(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    k: usize,
    n: usize,
    cols: usize,
) {
    // Scalar fallback for tail
    for j in 0..cols {
        for i in 0..RM {
            let mut sum = *c.add(i * n + j);
            for kk in 0..k {
                sum += *a.add(i * k + kk) * *b.add(kk * n + j);
            }
            *c.add(i * n + j) = sum;
        }
    }
}

/// Main SIMD GEMM function: C[M x N] += A[M x K] * B[K x N]
/// Uses 4x4 register blocking for ARM NEON
#[cfg(target_arch = "aarch64")]
pub unsafe fn simd_gemm(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    m: usize,
    k: usize,
    n: usize,
) {
    const KN: usize = GGML_F32_EPR;

    let mut ii = 0;
    let mut a_ptr = a;
    let mut c_ptr = c;

    // Process full RM blocks
    while ii + GEMM_RM <= m {
        let mut jj = 0;

        // Full RN blocks
        while jj + GEMM_RN * KN <= n {
            simd_gemm_ukernel::<GEMM_RM, GEMM_RN>(
                c_ptr.add(jj),
                a_ptr,
                b.add(jj),
                k,
                n,
            );
            jj += GEMM_RN * KN;
        }

        // Single KN blocks
        while jj + KN <= n {
            simd_gemm_ukernel::<GEMM_RM, 1>(
                c_ptr.add(jj),
                a_ptr,
                b.add(jj),
                k,
                n,
            );
            jj += KN;
        }

        // Tail columns
        if jj < n {
            simd_gemm_ukernel_tail::<GEMM_RM>(
                c_ptr.add(jj),
                a_ptr,
                b.add(jj),
                k,
                n,
                n - jj,
            );
        }

        a_ptr = a_ptr.add(GEMM_RM * k);
        c_ptr = c_ptr.add(GEMM_RM * n);
        ii += GEMM_RM;
    }

    // Tail rows: process one at a time
    while ii < m {
        let mut jj = 0;

        while jj + GEMM_RN * KN <= n {
            simd_gemm_ukernel::<1, GEMM_RN>(
                c_ptr.add(jj),
                a_ptr,
                b.add(jj),
                k,
                n,
            );
            jj += GEMM_RN * KN;
        }

        while jj + KN <= n {
            simd_gemm_ukernel::<1, 1>(c_ptr.add(jj), a_ptr, b.add(jj), k, n);
            jj += KN;
        }

        if jj < n {
            simd_gemm_ukernel_tail::<1>(
                c_ptr.add(jj),
                a_ptr,
                b.add(jj),
                k,
                n,
                n - jj,
            );
        }

        a_ptr = a_ptr.add(k);
        c_ptr = c_ptr.add(n);
        ii += 1;
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn simd_gemm(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    m: usize,
    k: usize,
    n: usize,
) {
    // Scalar fallback for non-ARM platforms
    for i in 0..m {
        for j in 0..n {
            let mut sum = *c.add(i * n + j);
            for kk in 0..k {
                sum += *a.add(i * k + kk) * *b.add(kk * n + j);
            }
            *c.add(i * n + j) = sum;
        }
    }
}
