// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

//! BLAS-accelerated GEMM for large batch sizes.
//!
//! Uses Apple's Accelerate framework (cblas_sgemm) for high-performance
//! matrix multiplication when batch size is large (n >= 32).

#[cfg(target_os = "macos")]
use std::os::raw::c_int;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    /// cblas_sgemm from Accelerate framework
    ///
    /// C := alpha * op(A) * op(B) + beta * C
    fn cblas_sgemm(
        order: c_int,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

/// BLAS constants
#[cfg(target_os = "macos")]
mod blas_const {
    pub const CblasRowMajor: i32 = 101;
    pub const CblasTrans: i32 = 112;
    pub const CblasNoTrans: i32 = 111;
}

/// Minimum batch size to trigger BLAS path
pub const BLAS_THRESHOLD: usize = 32;

/// Persistent BLAS context with reusable work buffer
///
/// Matches llama.cpp's ggml_backend_blas_context design
pub struct BlasContext {
    /// Work buffer for dequantized weights (persistent)
    work_data: Vec<f32>,
    /// Current buffer size in elements
    work_size: usize,
}

impl BlasContext {
    pub fn new() -> Self {
        Self { work_data: Vec::new(), work_size: 0 }
    }

    /// Ensure work buffer is large enough, reallocating only if needed
    pub fn ensure_size(&mut self, needed: usize) {
        if self.work_size < needed {
            // Reallocate with some headroom to reduce future reallocations
            let new_size = needed.max(self.work_size * 2);
            self.work_data.resize(new_size, 0.0);
            self.work_size = new_size;
        }
    }

    /// Get mutable slice of the work buffer
    pub fn as_mut_slice(&mut self, len: usize) -> &mut [f32] {
        self.ensure_size(len);
        &mut self.work_data[..len]
    }
}

impl Default for BlasContext {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local BLAS work buffer for reuse across layers
use std::cell::RefCell;
thread_local! {
    static BLAS_WORK: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

/// Execute FP32 GEMM using BLAS (Accelerate).
///
/// Computes: C[m,n] = A[m,k]^T * B[n,k]
///
/// # Safety
/// - All slices must have correct sizes
/// - m, n, k must be positive
#[cfg(target_os = "macos")]
#[inline]
pub unsafe fn gemm_blas_f32(
    c: *mut f32,
    a: *const f32,
    b: *const f32,
    m: usize,
    n: usize,
    k: usize,
) {
    use blas_const::*;

    // C := A^T * B
    // A is stored as [m, k] in row-major
    // With CblasTrans, op(A) = A^T which has shape [k, m]
    // B is stored as [n, k] in row-major
    // Result C has shape [k, n] -> but we want [m, n]
    //
    // Actually we want: C = A @ B^T
    // A is [m, k], B is [n, k] -> B^T is [k, n]
    // Result C is [m, n]
    //
    // Using CblasNoTrans for A and CblasTrans for B:
    // C = A @ B^T
    cblas_sgemm(
        CblasRowMajor, // Row-major order
        CblasNoTrans,  // No transpose A (A is [m,k])
        CblasTrans,    // Transpose B (B is [n,k], B^T is [k,n])
        m as c_int,    // Rows of A = m
        n as c_int,    // Columns of B^T = n
        k as c_int,    // Columns of A = Rows of B^T = k
        1.0,           // alpha
        a,             // A matrix
        k as c_int,    // LDA (leading dimension of A)
        b,             // B matrix
        k as c_int,    // LDB (leading dimension of B)
        0.0,           // beta
        c,             // C matrix
        n as c_int,    // LDC (leading dimension of C)
    );
}

/// Dequantize weights to FP32 for BLAS path.
///
/// Trait for converting quantized blocks to FP32 array.
pub trait DequantizeToFP32: Sized + Send + Sync {
    /// Number of values per quantization block
    fn values_per_block() -> usize;
    /// Dequantize source blocks to destination FP32 array
    fn dequantize_to_fp32(dst: &mut [f32], src: &[Self], k: usize);
}

/// Dequantize IQ4_XS to FP32
#[cfg(target_arch = "aarch64")]
impl DequantizeToFP32 for crate::quant::types::BlockIQ4XS {
    #[inline]
    fn values_per_block() -> usize {
        crate::quant::types::QK_K // 256
    }

    fn dequantize_to_fp32(
        dst: &mut [f32],
        src: &[crate::quant::types::BlockIQ4XS],
        k: usize,
    ) {
        use crate::quant::kv_cache::KVALUES_IQ4NL_I8;
        use crate::quant::types::QK_K;

        let nb = src.len();
        debug_assert_eq!(k, nb * QK_K);
        debug_assert_eq!(dst.len(), nb * QK_K);

        // Scalar implementation (can be vectorized later)
        for (i, block) in src.iter().enumerate() {
            let d = half::f16::from_bits(block.d).to_f32();
            let qs = &block.qs;
            let scales_l = &block.scales_l;
            let scales_h = block.scales_h;

            let base_idx = i * QK_K;

            // Process 4 sub-blocks (QK_K/64 = 4)
            for ib in 0..4 {
                // Extract scale for this sub-block
                let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) as i32
                    | ((((scales_h >> (2 * ib)) & 0x03) as i32) << 4);
                let dl = d * (ls - 32) as f32;

                // Process 32 values per sub-block
                let qs_offset = ib * 32;
                let dst_offset = base_idx + ib * 64;

                for j in 0..16 {
                    let q = qs[qs_offset + j];
                    dst[dst_offset + j] =
                        dl * KVALUES_IQ4NL_I8[(q & 0x0F) as usize] as f32;
                    dst[dst_offset + j + 16] =
                        dl * KVALUES_IQ4NL_I8[(q >> 4) as usize] as f32;
                }
            }
        }
    }
}

/// Dequantize Q4_K to FP32
#[cfg(target_arch = "aarch64")]
impl DequantizeToFP32 for crate::quant::types::BlockQ4K {
    #[inline]
    fn values_per_block() -> usize {
        crate::quant::types::QK_K // 256
    }

    fn dequantize_to_fp32(
        dst: &mut [f32],
        src: &[crate::quant::types::BlockQ4K],
        k: usize,
    ) {
        use crate::quant::types::QK_K;

        let nb = src.len();
        debug_assert_eq!(k, nb * QK_K);
        debug_assert_eq!(dst.len(), nb * QK_K);

        for (i, block) in src.iter().enumerate() {
            let d = half::f16::from_bits(block.d).to_f32();
            let min = half::f16::from_bits(block.dmin).to_f32();
            let qs = &block.qs;
            let scales = &block.scales;

            let base_idx = i * QK_K;
            let mut is = 0;

            // Process 4 sub-blocks of 64 values each
            for j in (0..QK_K).step_by(64) {
                // get_scale_min_k4 for scales[is] and scales[is+1]
                let (sc0, m0) = if is < 4 {
                    (scales[is] & 63, scales[is + 4] & 63)
                } else {
                    let d =
                        (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4);
                    let m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
                    (d, m)
                };

                let (sc1, m1) = if is + 1 < 4 {
                    (scales[is + 1] & 63, scales[is + 1 + 4] & 63)
                } else {
                    let d = (scales[is + 1 + 4] & 0x0F)
                        | ((scales[is + 1 - 4] >> 6) << 4);
                    let m = (scales[is + 1 + 4] >> 4)
                        | ((scales[is + 1] >> 6) << 4);
                    (d, m)
                };

                let d1 = d * sc0 as f32;
                let m1 = min * m0 as f32;
                let d2 = d * sc1 as f32;
                let m2 = min * m1 as f32;

                let qs_offset = (j / 64) * 32;
                let dst_offset = base_idx + j;

                // Lower 4 bits
                for l in 0..32 {
                    dst[dst_offset + l] =
                        d1 * (qs[qs_offset + l] & 0x0F) as f32 - m1;
                }
                // Upper 4 bits
                for l in 0..32 {
                    dst[dst_offset + 32 + l] =
                        d2 * (qs[qs_offset + l] >> 4) as f32 - m2;
                }

                is += 2;
            }
        }
    }
}

/// Dequantize Q5_K to FP32
#[cfg(target_arch = "aarch64")]
impl DequantizeToFP32 for crate::quant::types::BlockQ5K {
    #[inline]
    fn values_per_block() -> usize {
        crate::quant::types::QK_K // 256
    }

    fn dequantize_to_fp32(
        dst: &mut [f32],
        src: &[crate::quant::types::BlockQ5K],
        k: usize,
    ) {
        use crate::quant::types::QK_K;

        let nb = src.len();
        debug_assert_eq!(k, nb * QK_K);
        debug_assert_eq!(dst.len(), nb * QK_K);

        for (i, block) in src.iter().enumerate() {
            let d = half::f16::from_bits(block.d).to_f32();
            let min = half::f16::from_bits(block.dmin).to_f32();
            let ql = &block.qs;
            let qh = &block.qh;
            let scales = &block.scales;

            let base_idx = i * QK_K;
            let mut is = 0;
            let mut u1: u8 = 1;
            let mut u2: u8 = 2;

            // Process 4 sub-blocks of 64 values each
            for j in (0..QK_K).step_by(64) {
                // get_scale_min_k4 for scales[is] and scales[is+1]
                let (sc0, m0) = if is < 4 {
                    (scales[is] & 63, scales[is + 4] & 63)
                } else {
                    let d =
                        (scales[is + 4] & 0x0F) | ((scales[is - 4] >> 6) << 4);
                    let m = (scales[is + 4] >> 4) | ((scales[is] >> 6) << 4);
                    (d, m)
                };

                let (sc1, m1) = if is + 1 < 4 {
                    (scales[is + 1] & 63, scales[is + 1 + 4] & 63)
                } else {
                    let d = (scales[is + 1 + 4] & 0x0F)
                        | ((scales[is + 1 - 4] >> 6) << 4);
                    let m = (scales[is + 1 + 4] >> 4)
                        | ((scales[is + 1] >> 6) << 4);
                    (d, m)
                };

                let d1 = d * sc0 as f32;
                let m1 = min * m0 as f32;
                let d2 = d * sc1 as f32;
                let m2 = min * m1 as f32;

                let ql_offset = (j / 64) * 32;
                let dst_offset = base_idx + j;

                // Lower 4 bits + high bit
                for l in 0..32 {
                    let low = (ql[ql_offset + l] & 0x0F) as f32;
                    let high = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                    dst[dst_offset + l] = d1 * (low + high) - m1;
                }
                // Upper 4 bits + high bit
                for l in 0..32 {
                    let low = (ql[ql_offset + l] >> 4) as f32;
                    let high = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                    dst[dst_offset + 32 + l] = d2 * (low + high) - m2;
                }

                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub fn gemm_blas<T: DequantizeToFP32>(
    c: &mut [f32],
    a: &[T],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    threads: usize,
) {
    debug_assert_eq!(a.len() * T::values_per_block(), m * k);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);

    // Use persistent thread-local buffer
    let needed = m * k;
    let (a_f32_ptr, _len) = BLAS_WORK.with(|work| {
        let mut work = work.borrow_mut();
        if work.len() < needed {
            work.resize(needed, 0.0);
        }
        (work.as_mut_ptr(), work.len())
    });

    // Dequantize in parallel
    if threads > 1 {
        use crate::cpu::{SyncPtr, parallel_chunks};
        let k_per_block = T::values_per_block();
        let nb = k / k_per_block;
        let a_ptr = SyncPtr::new(a.as_ptr() as *mut T);
        let a_f32_ptr = SyncPtr::new(a_f32_ptr);
        unsafe {
            parallel_chunks(m, threads, move |start, end| {
                for i in start..end {
                    let row_a = a_ptr.get().add(i * nb);
                    let row_dst = a_f32_ptr.get().add(i * k);
                    T::dequantize_to_fp32(
                        std::slice::from_raw_parts_mut(row_dst, k),
                        std::slice::from_raw_parts(row_a, nb),
                        k,
                    );
                }
            });
        }
    } else {
        let nb = k / T::values_per_block();
        unsafe {
            for i in 0..m {
                let row_a = &a[i * nb..(i + 1) * nb];
                let row_dst =
                    std::slice::from_raw_parts_mut(a_f32_ptr.add(i * k), k);
                T::dequantize_to_fp32(row_dst, row_a, k);
            }
        }
    }

    // Call BLAS
    unsafe {
        gemm_blas_f32(c.as_mut_ptr(), a_f32_ptr, b.as_ptr(), m, n, k);
    }
}
