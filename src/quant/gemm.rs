// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

//! Unified GEMM template architecture for CPU inference.
//!
//! This module provides a clean, template-based GEMM implementation
//! matching llama.cpp's llamafile_sgemm architecture.
//!
//! # Design
//!
//! - `GemmKernel` trait: defines GEMM interface for each quantization format
//! - SIMD-optimized microkernel for FP32
//! - Dequantize-then-GEMM path for quantized formats
//! - Automatic BLAS path for large matrices (n >= 32)
//!
//! # Usage
//!
//! ```ignore
//! use crate::quant::gemm::gemm;
//!
//! // FP32 GEMM
//! gemm::<F32>(&mut c, &weights, &inputs, m, n, k, threads);
//!
//! // Q4_0 GEMM
//! gemm::<Q4_0>(&mut c, &weights, &inputs, m, n, k, threads);
//! ```

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod blas;
#[cfg(target_arch = "aarch64")]
mod fp32;
#[cfg(target_arch = "aarch64")]
mod iq4_nl;
#[cfg(target_arch = "aarch64")]
pub mod kernel;
#[cfg(target_arch = "aarch64")]
mod q4_0;
#[cfg(target_arch = "aarch64")]
mod q5_0;
#[cfg(target_arch = "aarch64")]
mod q8_0;
#[cfg(target_arch = "aarch64")]
pub mod scheduler;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub use blas::{BLAS_THRESHOLD, DequantizeToFP32};
#[cfg(target_arch = "aarch64")]
pub use scheduler::{
    BLAS_THRESHOLD_FLOPS, BLAS_THRESHOLD_N, GemmPath, WeightView,
    gemm_dispatch, gemm_dispatch_view, gemv_dispatch, requires_vec_dot,
    select_gemm_path, should_use_blas, supports_sgemm,
};

/// GEMM kernel trait for quantized formats.
///
/// Each quantization format implements this trait to provide
/// optimized matrix multiplication.
///
/// # Matrix Layout
///
/// All matrices are row-major:
/// - A (weights): [m, k] - m rows, k columns
/// - B (inputs): [n, k] - n rows, k columns
/// - C (output): [m, n] - m rows, n columns
///
/// This computes: C = A^T * B
#[cfg(target_arch = "aarch64")]
pub trait GemmKernel: Send + Sync {
    /// Weight storage type (e.g., BlockQ4_0 for Q4_0)
    type Weight: Send + Sync;

    /// Compute C[m,n] = A[m,k]^T * B[n,k]
    ///
    /// # Arguments
    ///
    /// * `c` - Output matrix [m, n] in row-major
    /// * `a` - Weight matrix [m, k] in row-major
    /// * `b` - Input matrix [n, k] in row-major (always FP32)
    /// * `m` - Number of output rows
    /// * `n` - Number of output columns (batch size)
    /// * `k` - Hidden dimension
    /// * `threads` - Number of threads to use
    fn gemm(
        c: &mut [f32],
        a: &[Self::Weight],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
        threads: usize,
    );
}

/// Unified GEMM entry point.
///
/// Automatically selects optimal path based on batch size:
/// - n >= 32: BLAS path (dequantize weights, then cblas_sgemm)
/// - n < 32: SIMD path (block tiling with on-the-fly dequantize)
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn gemm<T: GemmKernel>(
    c: &mut [f32],
    a: &[T::Weight],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    threads: usize,
) {
    T::gemm(c, a, b, m, n, k, threads);
}

/// FP32 type marker for GEMM kernel selection.
pub struct F32;

/// Q4_0 type marker for GEMM kernel selection.
#[allow(non_camel_case_types)]
pub struct Q4_0;

/// Q5_0 type marker for GEMM kernel selection.
#[allow(non_camel_case_types)]
pub struct Q5_0;

/// Q8_0 type marker for GEMM kernel selection.
#[allow(non_camel_case_types)]
pub struct Q8_0;

/// IQ4_NL type marker for GEMM kernel selection.
#[allow(non_camel_case_types)]
pub struct IQ4_NL;

// Non-aarch64 fallback
#[cfg(not(target_arch = "aarch64"))]
pub fn gemm_scalar(
    c: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[j * k + l];
            }
            c[j * m + i] = sum;
        }
    }
}
