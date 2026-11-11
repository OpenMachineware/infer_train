// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

//! Unified GEMM scheduler with 3-layer fallback.
//!
//! Matches llama.cpp's dispatch strategy:
//! 1. Layer 1: llamafile_sgemm (SIMD) - for non-K-quant formats
//! 2. Layer 2: vec_dot fallback - for K-quant formats
//! 3. Layer 3: BLAS - for large batches (n >= 32)
//!
//! # Dispatch Logic
//!
//! ```ignore
//! if format in [F32, Q4_0, Q5_0, Q8_0, IQ4_NL] {
//!     if n >= 32 && m * n * k > threshold {
//!         BLAS path (dequantize + cblas_sgemm)
//!     } else {
//!         SIMD path (on-the-fly dequantize)
//!     }
//! } else if format in [Q4_K, Q5_K, Q6_K, IQ4_XS, ...] {
//!     vec_dot path (quantize input to Q8_K, then vec_dot)
//! } else {
//!     fallback to scalar
//! }
//! ```

use crate::model::tensor::{DataType, Tensor};

/// Minimum batch size to trigger BLAS path
pub const BLAS_THRESHOLD_N: usize = 32;

/// Minimum FLOPs to trigger BLAS path (avoid BLAS overhead for small matrices)
pub const BLAS_THRESHOLD_FLOPS: usize = 1024 * 1024; // 1M FLOPs

/// Global BLAS enable flag (can be set at runtime)
static BLAS_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

/// Enable or disable BLAS path at runtime
pub fn set_blas_enabled(enabled: bool) {
    BLAS_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Check if BLAS is enabled
#[inline]
pub fn is_blas_enabled() -> bool {
    BLAS_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Check if BLAS path should be used
#[inline]
pub fn should_use_blas(m: usize, n: usize, k: usize) -> bool {
    is_blas_enabled()
        && n >= BLAS_THRESHOLD_N
        && m * n * k >= BLAS_THRESHOLD_FLOPS
}

/// Check if format supports llamafile_sgemm (Layer 1)
#[inline]
pub fn supports_sgemm(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::F32
            | DataType::F16
            | DataType::BF16
            | DataType::Q4_0
            | DataType::Q5_0
            | DataType::Q8_0
            | DataType::IQ4_NL
    )
}

/// Check if format requires vec_dot path (Layer 2)
#[inline]
pub fn requires_vec_dot(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::Q4_K
            | DataType::Q5_K
            | DataType::Q6_K
            | DataType::IQ4_XS
            | DataType::IQ3_XXS
            | DataType::IQ3_S
            | DataType::IQ2_XXS
            | DataType::IQ2_XS
            | DataType::IQ2_S
            | DataType::IQ1_S
    )
}

/// GEMM dispatch result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmPath {
    /// SIMD path (llamafile_sgemm style)
    Simd,
    /// BLAS path (dequantize + cblas_sgemm)
    Blas,
    /// vec_dot path (quantize input + vec_dot)
    VecDot,
    /// Scalar fallback
    Scalar,
}

/// Unified GEMM scheduler
///
/// Returns which path should be used for the given parameters
#[inline]
pub fn select_gemm_path(
    dtype: DataType,
    m: usize,
    n: usize,
    k: usize,
) -> GemmPath {
    if supports_sgemm(dtype) {
        // Non-K-quant: choose between SIMD and BLAS
        if should_use_blas(m, n, k) { GemmPath::Blas } else { GemmPath::Simd }
    } else if requires_vec_dot(dtype) {
        // K-quant: use BLAS for large batches, vec_dot for small
        if should_use_blas(m, n, k) { GemmPath::Blas } else { GemmPath::VecDot }
    } else {
        // Unknown format: scalar fallback
        GemmPath::Scalar
    }
}

/// Lightweight view of a weight tensor for dispatch.
/// Provides minimal interface needed by gemm_dispatch.
pub struct WeightView<'a> {
    pub dtype: DataType,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

impl<'a> WeightView<'a> {
    /// Create from raw components.
    pub fn new(dtype: DataType, shape: &'a [usize], bytes: &'a [u8]) -> Self {
        Self { dtype, shape, bytes }
    }

    /// Create from Tensor.
    pub fn from_tensor(tensor: &'a Tensor) -> Self {
        Self {
            dtype: tensor.dtype,
            shape: &tensor.shape,
            bytes: tensor.bytes(),
        }
    }
}

/// Execute GEMM with automatic path selection
///
/// # Arguments
/// * `c` - Output matrix [m, n]
/// * `a` - Weight matrix [m, k] (quantized)
/// * `b` - Input matrix [n, k] (FP32)
/// * `m` - Number of output rows
/// * `n` - Number of output columns (batch size)
/// * `k` - Hidden dimension
/// * `threads` - Number of threads
/// * `q8k_buf` - Q8_K buffer for vec_dot path (optional)
///
/// # Returns
/// true if successful, false if format not supported
#[cfg(target_arch = "aarch64")]
pub fn gemm_dispatch(
    c: &mut [f32],
    weights: &Tensor,
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    threads: usize,
    q8k_buf: Option<&mut [u8]>,
) -> bool {
    let view = WeightView::from_tensor(weights);
    gemm_dispatch_view(c, &view, b, m, n, k, threads, q8k_buf)
}

/// Execute GEMM with WeightView (for Backend integration).
#[cfg(target_arch = "aarch64")]
pub fn gemm_dispatch_view(
    c: &mut [f32],
    weights: &WeightView,
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    threads: usize,
    q8k_buf: Option<&mut [u8]>,
) -> bool {
    let dtype = weights.dtype;
    let path = select_gemm_path(dtype, m, n, k);

    match path {
        GemmPath::Simd | GemmPath::Blas => {
            // Use GemmKernel for non-K-quant
            use crate::quant::gemm::{F32, IQ4_NL, Q4_0, Q5_0, Q8_0, gemm};

            match dtype {
                DataType::F32 => {
                    let a: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            weights.bytes.as_ptr() as *const f32,
                            m * k,
                        )
                    };
                    gemm::<F32>(c, a, b, m, n, k, threads);
                    true
                }
                DataType::Q4_0 => {
                    use crate::quant::types::BlockQ4_0;
                    let a: &[BlockQ4_0] = unsafe {
                        std::slice::from_raw_parts(
                            weights.bytes.as_ptr() as *const BlockQ4_0,
                            m * (k / 32),
                        )
                    };
                    gemm::<Q4_0>(c, a, b, m, n, k, threads);
                    true
                }
                DataType::Q5_0 => {
                    use crate::quant::types::BlockQ5_0;
                    let a: &[BlockQ5_0] = unsafe {
                        std::slice::from_raw_parts(
                            weights.bytes.as_ptr() as *const BlockQ5_0,
                            m * (k / 32),
                        )
                    };
                    gemm::<Q5_0>(c, a, b, m, n, k, threads);
                    true
                }
                DataType::Q8_0 => {
                    use crate::quant::types::BlockQ8_0;
                    let a: &[BlockQ8_0] = unsafe {
                        std::slice::from_raw_parts(
                            weights.bytes.as_ptr() as *const BlockQ8_0,
                            m * (k / 32),
                        )
                    };
                    gemm::<Q8_0>(c, a, b, m, n, k, threads);
                    true
                }
                DataType::IQ4_NL => {
                    use crate::quant::types::BlockIQ4NL;
                    let a: &[BlockIQ4NL] = unsafe {
                        std::slice::from_raw_parts(
                            weights.bytes.as_ptr() as *const BlockIQ4NL,
                            m * (k / 32),
                        )
                    };
                    gemm::<IQ4_NL>(c, a, b, m, n, k, threads);
                    true
                }
                // K-quant with BLAS path: dequantize to F32 then call cblas_sgemm
                DataType::IQ4_XS => {
                    use crate::quant::gemm::blas::gemm_blas;
                    use crate::quant::types::{BlockIQ4XS, QK_K};

                    let nb = k / QK_K;
                    let a: &[BlockIQ4XS] = unsafe {
                        std::slice::from_raw_parts(
                            weights.bytes.as_ptr() as *const BlockIQ4XS,
                            m * nb,
                        )
                    };
                    gemm_blas(c, a, b, m, n, k, threads);
                    true
                }
                DataType::Q4_K => {
                    use crate::quant::gemm::blas::gemm_blas;
                    use crate::quant::types::{BlockQ4K, QK_K};

                    let nb = k / QK_K;
                    let a: &[BlockQ4K] = unsafe {
                        std::slice::from_raw_parts(
                            weights.bytes.as_ptr() as *const BlockQ4K,
                            m * nb,
                        )
                    };
                    gemm_blas(c, a, b, m, n, k, threads);
                    true
                }
                DataType::Q5_K => {
                    use crate::quant::gemm::blas::gemm_blas;
                    use crate::quant::types::{BlockQ5K, QK_K};

                    let nb = k / QK_K;
                    let a: &[BlockQ5K] = unsafe {
                        std::slice::from_raw_parts(
                            weights.bytes.as_ptr() as *const BlockQ5K,
                            m * nb,
                        )
                    };
                    gemm_blas(c, a, b, m, n, k, threads);
                    true
                }
                _ => false,
            }
        }
        GemmPath::VecDot => {
            // K-quant: use existing matmul
            use crate::quant::types::QK_K;
            let k_padded = ((k + QK_K - 1) / QK_K) * QK_K;
            let nb = k_padded / QK_K;

            // Need Q8_K buffer
            let q8k_buf = match q8k_buf {
                Some(buf) => buf,
                None => return false,
            };

            // Quantize input to Q8_K
            use crate::quant::kv_cache::{
                quantize_row_q8_k_neon, quantize_row_q8_k_parallel_raw,
            };
            use crate::quant::types::BlockQ8K;

            // For batch prefill (n >= 32), use parallel quantization
            // For decode (n == 1), use single-thread to avoid barrier overhead
            let x_q8k_blocks = unsafe {
                let available = q8k_buf.len() / std::mem::size_of::<BlockQ8K>();
                if available < n * nb {
                    return false;
                }
                std::slice::from_raw_parts_mut(
                    q8k_buf.as_mut_ptr() as *mut BlockQ8K,
                    n * nb,
                )
            };

            if n >= 32 {
                // Multi-threaded parallel quantization (matching llama.cpp)
                // Prepare input slices for parallel quantization
                let inputs: Vec<&[f32]> =
                    (0..n).map(|i| &b[i * k..(i + 1) * k]).collect();

                // Parallel quantization with barrier (pass raw pointer)
                use crate::cpu::SyncPtr;
                let x_q8k_ptr = SyncPtr::new(x_q8k_blocks.as_mut_ptr());
                quantize_row_q8_k_parallel_raw(
                    &inputs, x_q8k_ptr, n, nb, threads,
                );
            } else {
                // Single-thread quantization for small batches
                if k != k_padded {
                    // Would need padding buffer
                    return false;
                }
                // Quantize each row
                for i in 0..n {
                    let input_row = &b[i * k..(i + 1) * k];
                    let output_blocks = &mut x_q8k_blocks[i * nb..(i + 1) * nb];
                    quantize_row_q8_k_neon(input_row, output_blocks);
                }
            }

            // Call appropriate K-quant matmul
            use crate::quant::matmul::k_quant::{
                matmul_iq4_xs_q8_k_hybrid_with_threads,
                matmul_q4_k_q8_k_hybrid_with_threads,
                matmul_q5_k_q8_k_hybrid_with_threads,
                matmul_q6_k_q8_k_hybrid_with_threads,
            };
            use crate::quant::types::{
                BlockIQ4XS, BlockQ4K, BlockQ5K, BlockQ6K,
            };

            unsafe {
                match dtype {
                    DataType::Q4_K => {
                        let a: &[BlockQ4K] = std::slice::from_raw_parts(
                            weights.bytes.as_ptr() as *const BlockQ4K,
                            m * nb,
                        );
                        matmul_q4_k_q8_k_hybrid_with_threads(
                            a,
                            x_q8k_blocks,
                            c,
                            k_padded,
                            m,
                            threads,
                        );
                    }
                    DataType::Q5_K => {
                        let a: &[BlockQ5K] = std::slice::from_raw_parts(
                            weights.bytes.as_ptr() as *const BlockQ5K,
                            m * nb,
                        );
                        matmul_q5_k_q8_k_hybrid_with_threads(
                            a,
                            x_q8k_blocks,
                            c,
                            k_padded,
                            m,
                            threads,
                        );
                    }
                    DataType::Q6_K => {
                        let a: &[BlockQ6K] = std::slice::from_raw_parts(
                            weights.bytes.as_ptr() as *const BlockQ6K,
                            m * nb,
                        );
                        matmul_q6_k_q8_k_hybrid_with_threads(
                            a,
                            x_q8k_blocks,
                            c,
                            k_padded,
                            m,
                            threads,
                        );
                    }
                    DataType::IQ4_XS => {
                        let a: &[BlockIQ4XS] = std::slice::from_raw_parts(
                            weights.bytes.as_ptr() as *const BlockIQ4XS,
                            m * nb,
                        );
                        matmul_iq4_xs_q8_k_hybrid_with_threads(
                            a,
                            x_q8k_blocks,
                            c,
                            k_padded,
                            m,
                            threads,
                        );
                    }
                    _ => return false,
                }
            }
            true
        }
        GemmPath::Scalar => {
            // Fallback: not implemented
            false
        }
    }
}

/// Execute GEMV (matrix-vector multiplication) with automatic path selection
///
/// This is a convenience wrapper around gemm_dispatch for single vector input.
/// Equivalent to gemm_dispatch with n=1.
///
/// # Arguments
/// * `out` - Output vector [m]
/// * `weights` - Weight matrix [m, k] (quantized)
/// * `x` - Input vector [k] (FP32)
/// * `threads` - Number of threads
/// * `q8k_buf` - Q8_K buffer for vec_dot path (optional)
/// * `x_padded` - Padding buffer for non-standard k (optional)
#[cfg(target_arch = "aarch64")]
pub fn gemv_dispatch(
    out: &mut [f32],
    weights: &Tensor,
    x: &[f32],
    threads: usize,
    q8k_buf: Option<&mut [u8]>,
    x_padded: Option<&mut [f32]>,
) -> bool {
    let k = x.len();
    let m = out.len();

    // Get weight shape
    let w_shape = &weights.shape;
    if w_shape.len() != 2 {
        eprintln!("[ERROR] Weight tensor must be 2D, got {}D", w_shape.len());
        return false;
    }

    // Weight is [m, k] in row-major
    let w_m = w_shape[0];
    let w_k = w_shape[1];

    if w_m != m || w_k != k {
        eprintln!(
            "[ERROR] Shape mismatch in GEMV: tensor name='{}', weight [{}, {}], output [{}], input [{}]",
            weights.name, w_m, w_k, m, k
        );
        return false;
    }

    // For K-quant formats with non-standard k, handle padding
    if requires_vec_dot(weights.dtype) {
        use crate::quant::types::QK_K;
        let k_padded = ((k + QK_K - 1) / QK_K) * QK_K;

        if k != k_padded {
            // Need padding buffer
            let x_padded = match x_padded {
                Some(buf) => buf,
                None => {
                    eprintln!(
                        "[ERROR] Padding buffer required for k={} (padded to {})",
                        k, k_padded
                    );
                    return false;
                }
            };

            // Zero padding area
            for i in k..k_padded {
                x_padded[i] = 0.0;
            }
            x_padded[..k].copy_from_slice(x);

            // Call with padded input
            let mut padded_out = out.to_vec();
            let result = gemm_dispatch(
                &mut padded_out,
                weights,
                x_padded,
                m,
                1,
                k_padded,
                threads,
                q8k_buf,
            );
            if result {
                out.copy_from_slice(&padded_out);
            }
            return result;
        }
    }

    // Call gemm_dispatch with n=1
    gemm_dispatch(out, weights, x, m, 1, k, threads, q8k_buf)
}

#[cfg(not(target_arch = "aarch64"))]
pub fn gemv_dispatch(
    _out: &mut [f32],
    _weights: &Tensor,
    _x: &[f32],
    _threads: usize,
    _q8k_buf: Option<&mut [u8]>,
    _x_padded: Option<&mut [f32]>,
) -> bool {
    false
}
