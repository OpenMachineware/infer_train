// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use std::arch::aarch64::*;

/// FP32 GEMM: C = A^T * B
///
/// - A: [m, k] in row-major (lda = k)
/// - B: [n, k] in row-major (ldb = k)
/// - C: [n, m] in row-major (ldc = m)
///
/// Uses tinyBLAS approach from llama.cpp: block tiling with FMLA
#[target_feature(enable = "neon")]
pub unsafe fn sgemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    // For small n, use simple row-by-row approach
    if n < 4 {
        for j in 0..n {
            for i in 0..m {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[j * k + l];
                }
                c[j * m + i] = sum;
            }
        }
        return;
    }

    // Block tiling parameters for ARM NEON (16 registers)
    // RM = 4 (rows per block), RN = 3 (cols per block)
    // This uses 4*3 + 4 + 3 = 19 > 16 registers, so we need careful scheduling
    // Actually llama.cpp uses RN=3 with accumulator reuse

    const RM: usize = 4; // Process 4 rows at a time
    const RN: usize = 3; // Process 3 columns at a time
    const KN: usize = 4; // 4 floats per float32x4_t

    // Process in blocks of RM rows
    for i0 in (0..m).step_by(RM) {
        let i1 = (i0 + RM).min(m);

        // Process in blocks of RN columns
        for j0 in (0..n).step_by(RN) {
            let j1 = (j0 + RN).min(n);

            // Compute block C[j0:j1, i0:i1] = A[i0:i1, :] * B[j0:j1, :]^T
            gemm_bloc(a, b, c, m, k, i0, i1, j0, j1);
        }
    }
}

/// Compute a single block of the GEMM
#[target_feature(enable = "neon")]
#[inline]
unsafe fn gemm_bloc(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ldc: usize,
    k: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    let rm = i1 - i0;
    let rn = j1 - j0;

    // Accumulators for each output element
    // C[j][i] for j in [j0, j1), i in [i0, i1)
    let mut cv = [[0.0f32; 4]; 3]; // cv[col][row]

    // Process k in chunks of 4 (KN)
    // Main loop: process 4 elements of k at a time
    let k4 = k - k % 4;
    for l in (0..k4).step_by(4) {
        // Load 4 rows from A
        let av = [
            if i0 + 0 < i1 {
                vld1q_f32(a.as_ptr().add((i0 + 0) * k + l))
            } else {
                vdupq_n_f32(0.0)
            },
            if i0 + 1 < i1 {
                vld1q_f32(a.as_ptr().add((i0 + 1) * k + l))
            } else {
                vdupq_n_f32(0.0)
            },
            if i0 + 2 < i1 {
                vld1q_f32(a.as_ptr().add((i0 + 2) * k + l))
            } else {
                vdupq_n_f32(0.0)
            },
            if i0 + 3 < i1 {
                vld1q_f32(a.as_ptr().add((i0 + 3) * k + l))
            } else {
                vdupq_n_f32(0.0)
            },
        ];

        // For each column j in [j0, j1), load from B and do FMLA
        for j in 0..rn {
            let bv = vld1q_f32(b.as_ptr().add((j0 + j) * k + l));
            // Fused multiply-add: cv[j][i] += av[i] * bv
            cv[j][0] =
                vaddvq_f32(vfmaq_f32(vdupq_n_f32(0.0), av[0], bv)) + cv[j][0];
            cv[j][1] =
                vaddvq_f32(vfmaq_f32(vdupq_n_f32(0.0), av[1], bv)) + cv[j][1];
            cv[j][2] =
                vaddvq_f32(vfmaq_f32(vdupq_n_f32(0.0), av[2], bv)) + cv[j][2];
            cv[j][3] =
                vaddvq_f32(vfmaq_f32(vdupq_n_f32(0.0), av[3], bv)) + cv[j][3];
        }
    }

    // Handle remaining k elements
    for l in k4..k {
        for j in 0..rn {
            let bval = b[(j0 + j) * k + l];
            for i in 0..rm {
                cv[j][i] += a[(i0 + i) * k + l] * bval;
            }
        }
    }

    // Store results
    for j in 0..rn {
        for i in 0..rm {
            c[(j0 + j) * ldc + (i0 + i)] = cv[j][i];
        }
    }
}

/// Optimized FP32 GEMM with block reordering
/// Uses register blocking for better ILP
#[target_feature(enable = "neon")]
pub unsafe fn sgemm_optimized(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    if n < 4 {
        sgemm(a, b, c, m, n, k);
        return;
    }

    // llama.cpp's tinyBLAS uses accumulator vectors that persist across k iterations
    // This is the key optimization: accumulate in SIMD registers

    const RM: usize = 4;
    const RN: usize = 3;

    for i0 in (0..m).step_by(RM) {
        let i1 = (i0 + RM).min(m);

        for j0 in (0..n).step_by(RN) {
            let j1 = (j0 + RN).min(n);
            gemm_bloc_optimized(a, b, c, m, k, i0, i1, j0, j1);
        }
    }
}

/// Optimized block computation using SIMD accumulators
/// Key: hoist boundary checks OUT of K-loop
#[target_feature(enable = "neon")]
#[inline(never)]
unsafe fn gemm_bloc_optimized(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ldc: usize,
    k: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    // Pre-compute A row pointers (hoist boundary checks out of K-loop)
    // This is the critical optimization - C++ does this implicitly
    let a0 = if i0 + 0 < i1 {
        a.as_ptr().add((i0 + 0) * k)
    } else {
        std::ptr::NonNull::dangling().as_ptr()
    };
    let a1 = if i0 + 1 < i1 {
        a.as_ptr().add((i0 + 1) * k)
    } else {
        std::ptr::NonNull::dangling().as_ptr()
    };
    let a2 = if i0 + 2 < i1 {
        a.as_ptr().add((i0 + 2) * k)
    } else {
        std::ptr::NonNull::dangling().as_ptr()
    };
    let a3 = if i0 + 3 < i1 {
        a.as_ptr().add((i0 + 3) * k)
    } else {
        std::ptr::NonNull::dangling().as_ptr()
    };

    // Pre-compute valid row mask
    let row_mask = [i0 + 0 < i1, i0 + 1 < i1, i0 + 2 < i1, i0 + 3 < i1];

    // Pre-compute B column pointers
    let b0 = b.as_ptr().add(j0 * k);
    let b1 = if j0 + 1 < j1 {
        b.as_ptr().add((j0 + 1) * k)
    } else {
        std::ptr::NonNull::dangling().as_ptr()
    };
    let b2 = if j0 + 2 < j1 {
        b.as_ptr().add((j0 + 2) * k)
    } else {
        std::ptr::NonNull::dangling().as_ptr()
    };

    let col_mask = [
        true,        // j0 always valid
        j0 + 1 < j1, // j1 valid?
        j0 + 2 < j1, // j2 valid?
    ];

    // Initialize accumulators - use fixed 3x4 layout matching llama.cpp
    let mut cv0 = [
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
    ]; // col 0
    let mut cv1 = [
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
    ]; // col 1
    let mut cv2 = [
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
        vdupq_n_f32(0.0),
    ]; // col 2

    // Main K-loop: NO boundary checks inside
    let k4 = k - k % 4;
    for l in (0..k4).step_by(4) {
        // Load from A (may load garbage for invalid rows, but won't be used)
        let av = [
            vld1q_f32(a0.add(l)),
            vld1q_f32(a1.add(l)),
            vld1q_f32(a2.add(l)),
            vld1q_f32(a3.add(l)),
        ];

        // Load from B and FMLA - fixed 3 columns
        // Only accumulate to valid positions
        if col_mask[0] {
            let bv = vld1q_f32(b0.add(l));
            cv0[0] = vfmaq_f32(cv0[0], av[0], bv);
            cv0[1] = vfmaq_f32(cv0[1], av[1], bv);
            cv0[2] = vfmaq_f32(cv0[2], av[2], bv);
            cv0[3] = vfmaq_f32(cv0[3], av[3], bv);
        }
        if col_mask[1] {
            let bv = vld1q_f32(b1.add(l));
            cv1[0] = vfmaq_f32(cv1[0], av[0], bv);
            cv1[1] = vfmaq_f32(cv1[1], av[1], bv);
            cv1[2] = vfmaq_f32(cv1[2], av[2], bv);
            cv1[3] = vfmaq_f32(cv1[3], av[3], bv);
        }
        if col_mask[2] {
            let bv = vld1q_f32(b2.add(l));
            cv2[0] = vfmaq_f32(cv2[0], av[0], bv);
            cv2[1] = vfmaq_f32(cv2[1], av[1], bv);
            cv2[2] = vfmaq_f32(cv2[2], av[2], bv);
            cv2[3] = vfmaq_f32(cv2[3], av[3], bv);
        }
    }

    // Horizontal sum and store - only for valid rows/cols
    let a_ptrs = [a0, a1, a2, a3];
    let cvs = [cv0, cv1, cv2];
    for j in 0..3 {
        if !col_mask[j] {
            continue;
        }
        for i in 0..4 {
            if !row_mask[i] {
                continue;
            }
            let sum = vaddvq_f32(cvs[j][i]);
            let mut tail = 0.0f32;
            for l in k4..k {
                tail += *a_ptrs[i].add(l) * *b.as_ptr().add((j0 + j) * k + l);
            }
            c[(j0 + j) * ldc + (i0 + i)] = sum + tail;
        }
    }
}
