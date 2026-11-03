// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#![allow(asm_sub_register)]

use crate::quant::types::*;
use crate::quant::vec_dot::arm::*;
use once_cell::sync::Lazy;
use std::arch::aarch64::*;
use std::arch::asm;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Cache for interleaved Q4_K weights
/// Key: pointer to original weights
/// Value: interleaved weights (m/8 blocks of BlockQ4Kx8)
static Q4K_INTERLEAVED_CACHE: Lazy<
    Arc<Mutex<HashMap<usize, Vec<BlockQ4Kx8>>>>,
> = Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

/// Manual SDOT implementation via inline asm
#[allow(dead_code)]
#[inline(always)]
unsafe fn vdotq_s32_manual(
    acc: int32x4_t,
    a: int8x16_t,
    b: int8x16_t,
) -> int32x4_t {
    let mut result = acc;
    asm!(
        "sdot {0}.4s, {1}.16b, {2}.16b",
        inout(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, preserves_flags)
    );
    result
}

/// Optimized Q4_K × Q8_K matrix-vector multiplication
/// Uses chunk-based parallel execution for cache-friendly memory access
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q4_k_q8_k(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;
    // Use standard parallel kernel - interleaved kernel is not parallelized
    matmul_q4_k_q8_k_standard(weights, x, dst, k, m, nb);
}

/// Standard Q4_K × Q8_K matrix-vector multiplication
/// Uses std::thread::scope for low-overhead parallel execution
#[target_feature(enable = "neon,dotprod")]
unsafe fn matmul_q4_k_q8_k_standard(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
    nb: usize,
) {
    use crate::cpu::{SyncPtr, default_n_threads, parallel_chunks};

    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
    let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ4K);
    let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8K);

    parallel_chunks(m, default_n_threads(), move |start, end| {
        for row in start..end {
            *dst_ptr.get().add(row) = vec_dot_q4_k_q8_k_neon(
                k,
                std::slice::from_raw_parts(weights_ptr.get().add(row * nb), nb),
                std::slice::from_raw_parts(x_ptr.get(), nb),
            );
        }
    });
}

/// Single-thread Q4_K × Q8_K matrix-vector multiplication
/// For decode (batch=1) where thread overhead dominates
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q4_k_q8_k_single_thread(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for row in 0..m {
        dst[row] = vec_dot_q4_k_q8_k_neon(
            k,
            std::slice::from_raw_parts(weights.as_ptr().add(row * nb), nb),
            x,
        );
    }
}

/// Wrapper function for lock-free thread pool
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemv_q4_k_q8_k_kernel(
    weights: *const u8,
    input: *const u8,
    output: *mut f32,
    k: usize,
    m: usize,
    start: usize,
    end: usize,
) {
    let nb = k / QK_K;
    let weights = weights as *const BlockQ4K;
    let input = input as *const BlockQ8K;

    for row in start..end {
        *output.add(row) = vec_dot_q4_k_q8_k_neon(
            k,
            std::slice::from_raw_parts(weights.add(row * nb), nb),
            std::slice::from_raw_parts(input, nb),
        );
    }
}

/// Persistent thread pool Q4_K × Q8_K matrix-vector multiplication
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q4_k_q8_k_persistent(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let pool = crate::cpu::global_pool();
    pool.execute_gemv(
        gemv_q4_k_q8_k_kernel,
        weights.as_ptr() as *const u8,
        x.as_ptr() as *const u8,
        dst.as_mut_ptr(),
        k,
        m,
    );
}

/// Hybrid Q4_K × Q8_K: ST for single row, MT with persistent pool for m > 1
/// Key insight from llama.cpp: always use threads for parallelizable work
/// Persistent thread pool has ~590ns overhead (95x faster than std::thread::scope)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q4_k_q8_k_hybrid(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    if m <= 1 {
        matmul_q4_k_q8_k_single_thread(weights, x, dst, k, m);
    } else {
        matmul_q4_k_q8_k_persistent(weights, x, dst, k, m);
    }
}

/// Q6_K × Q8_K matrix-vector multiplication
/// Uses std::thread::scope for low-overhead parallel execution
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q6_k_q8_k(
    weights: &[BlockQ6K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    use crate::cpu::{SyncPtr, default_n_threads, parallel_chunks};

    let nb = k / QK_K;
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
    let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ6K);
    let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8K);

    parallel_chunks(m, default_n_threads(), move |start, end| {
        for row in start..end {
            *dst_ptr.get().add(row) = vec_dot_q6_k_q8_k_neon(
                k,
                std::slice::from_raw_parts(weights_ptr.get().add(row * nb), nb),
                std::slice::from_raw_parts(x_ptr.get(), nb),
            );
        }
    });
}

/// Single-thread Q6_K × Q8_K matrix-vector multiplication
/// For decode (batch=1) where thread overhead dominates
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q6_k_q8_k_single_thread(
    weights: &[BlockQ6K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for row in 0..m {
        dst[row] = vec_dot_q6_k_q8_k_neon(
            k,
            std::slice::from_raw_parts(weights.as_ptr().add(row * nb), nb),
            x,
        );
    }
}

/// Wrapper function for lock-free thread pool
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemv_q6_k_q8_k_kernel(
    weights: *const u8,
    input: *const u8,
    output: *mut f32,
    k: usize,
    m: usize,
    start: usize,
    end: usize,
) {
    let nb = k / QK_K;
    let weights = weights as *const BlockQ6K;
    let input = input as *const BlockQ8K;

    for row in start..end {
        *output.add(row) = vec_dot_q6_k_q8_k_neon(
            k,
            std::slice::from_raw_parts(weights.add(row * nb), nb),
            std::slice::from_raw_parts(input, nb),
        );
    }
}

/// Persistent thread pool Q6_K × Q8_K matrix-vector multiplication
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q6_k_q8_k_persistent(
    weights: &[BlockQ6K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let pool = crate::cpu::global_pool();
    pool.execute_gemv(
        gemv_q6_k_q8_k_kernel,
        weights.as_ptr() as *const u8,
        x.as_ptr() as *const u8,
        dst.as_mut_ptr(),
        k,
        m,
    );
}

/// Hybrid Q6_K × Q8_K: ST for single row, MT with persistent pool for m > 1
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q6_k_q8_k_hybrid(
    weights: &[BlockQ6K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    if m <= 1 {
        matmul_q6_k_q8_k_single_thread(weights, x, dst, k, m);
    } else {
        matmul_q6_k_q8_k_persistent(weights, x, dst, k, m);
    }
}

/// Q5_K × Q8_K matrix-vector multiplication
/// Uses std::thread::scope for low-overhead parallel execution
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q5_k_q8_k(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    use crate::cpu::{SyncPtr, default_n_threads, parallel_chunks};

    let nb = k / QK_K;
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
    let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ5K);
    let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8K);

    parallel_chunks(m, default_n_threads(), move |start, end| {
        for row in start..end {
            *dst_ptr.get().add(row) = vec_dot_q5_k_q8_k_neon(
                k,
                std::slice::from_raw_parts(weights_ptr.get().add(row * nb), nb),
                std::slice::from_raw_parts(x_ptr.get(), nb),
            );
        }
    });
}

/// Single-thread Q5_K × Q8_K matrix-vector multiplication
/// For decode (batch=1) where thread overhead dominates
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q5_k_q8_k_single_thread(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    // Narrow path optimization for decode (m=1, typical hidden_size)
    // Process multiple K blocks together to fit in L1 cache
    // L1 = 31KB, so we can fit ~120 blocks of Q5_K (256 bytes each)
    // Use k_extent = 31KB / (16 + m) floored to 256 multiples
    let l1_size = 31 * 1024;
    let rows_for_cache = if m < 16 { 16 } else { m };
    let mut k_extent_bytes = l1_size / (16 + rows_for_cache);
    k_extent_bytes = k_extent_bytes / QK_K * QK_K; // Floor to QK_K multiple

    let narrow_k_blocks =
        if k_extent_bytes > 0 { (k_extent_bytes / QK_K).min(nb) } else { nb };

    // If narrow path is beneficial, use it
    if narrow_k_blocks >= 4 && m == 1 {
        // Narrow path: process multiple K blocks at once for better L1 utilization
        // Each weight row is read in larger chunks
        for row in 0..m {
            let mut sum = 0.0f32;
            let w_row = &weights[row * nb..(row + 1) * nb];

            // Process K in chunks of narrow_k_blocks
            for chunk_start in (0..nb).step_by(narrow_k_blocks) {
                let chunk_end = (chunk_start + narrow_k_blocks).min(nb);

                // Process this K chunk
                for b in chunk_start..chunk_end {
                    sum += vec_dot_q5_k_q8_k_neon(
                        QK_K,
                        std::slice::from_raw_parts(w_row.as_ptr().add(b), 1),
                        &x[b..b + 1],
                    );
                }
            }
            dst[row] = sum;
        }
    } else {
        // Standard path: one vec_dot per row
        for row in 0..m {
            dst[row] = vec_dot_q5_k_q8_k_neon(
                k,
                std::slice::from_raw_parts(weights.as_ptr().add(row * nb), nb),
                x,
            );
        }
    }
}

/// Single-thread Q5_K × Q8_K with narrow path optimization
/// Processes K dimension in larger chunks for better L1 cache utilization
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q5_k_q8_k_narrow(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    // Calculate optimal K extent for L1 cache
    // L1 = 31KB (leaving room for scales and code)
    // Each Q5_K block is 444 bytes (but we read in 256-element chunks)
    // For m rows, we need: m * k_extent * sizeof(weight_elem)
    // k_extent = L1 / (m * weight_size)
    let l1_size = 31 * 1024;
    let weight_bytes_per_k = 1; // 1 byte per element after unpacking
    let k_extent = l1_size / (m * weight_bytes_per_k);
    let narrow_k_blocks = (k_extent / QK_K).min(nb).max(1);

    for row in 0..m {
        let mut sum = 0.0f32;
        let w_row = &weights[row * nb..(row + 1) * nb];

        // Process K in larger chunks
        for chunk_start in (0..nb).step_by(narrow_k_blocks) {
            let chunk_end = (chunk_start + narrow_k_blocks).min(nb);

            // Inner loop: multiple blocks for better prefetching
            for b in chunk_start..chunk_end {
                sum += vec_dot_q5_k_q8_k_neon(
                    QK_K,
                    std::slice::from_raw_parts(w_row.as_ptr().add(b), 1),
                    &x[b..b + 1],
                );
            }
        }
        dst[row] = sum;
    }
}

/// Wrapper function for lock-free thread pool
/// Matches the signature: fn(*const u8, *const u8, *mut f32, usize, usize, usize, usize)
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemv_q5_k_q8_k_kernel(
    weights: *const u8,
    input: *const u8,
    output: *mut f32,
    k: usize,
    m: usize,
    start: usize,
    end: usize,
) {
    let nb = k / QK_K;
    let weights = weights as *const BlockQ5K;
    let input = input as *const BlockQ8K;

    for row in start..end {
        *output.add(row) = vec_dot_q5_k_q8_k_neon(
            k,
            std::slice::from_raw_parts(weights.add(row * nb), nb),
            std::slice::from_raw_parts(input, nb),
        );
    }
}

/// Persistent thread pool Q5_K × Q8_K matrix-vector multiplication
/// Zero thread spawn overhead - uses global persistent thread pool
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q5_k_q8_k_persistent(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let pool = crate::cpu::global_pool();
    pool.execute_gemv(
        gemv_q5_k_q8_k_kernel,
        weights.as_ptr() as *const u8,
        x.as_ptr() as *const u8,
        dst.as_mut_ptr(),
        k,
        m,
    );
}

/// Hybrid Q5_K × Q8_K: ST for single row, MT with persistent pool for m > 1
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q5_k_q8_k_hybrid(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    // Persistent pool has ~590ns overhead, use for any m > 1
    if m <= 1 {
        matmul_q5_k_q8_k_single_thread(weights, x, dst, k, m);
    } else {
        matmul_q5_k_q8_k_persistent(weights, x, dst, k, m);
    }
}

/// Q3_K × Q8_K matrix-vector multiplication
/// Uses std::thread::scope for low-overhead parallel execution
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q3_k_q8_k(
    weights: &[BlockQ3K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    use crate::cpu::{SyncPtr, default_n_threads, parallel_chunks};

    let nb = k / QK_K;
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
    let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ3K);
    let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8K);

    parallel_chunks(m, default_n_threads(), move |start, end| {
        for row in start..end {
            *dst_ptr.get().add(row) = vec_dot_q3_k_q8_k_neon(
                k,
                std::slice::from_raw_parts(weights_ptr.get().add(row * nb), nb),
                std::slice::from_raw_parts(x_ptr.get(), nb),
            );
        }
    });
}

/// Single-thread Q3_K × Q8_K matrix-vector multiplication
/// For decode (batch=1) where thread overhead dominates
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q3_k_q8_k_single_thread(
    weights: &[BlockQ3K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for row in 0..m {
        dst[row] = vec_dot_q3_k_q8_k_neon(
            k,
            std::slice::from_raw_parts(weights.as_ptr().add(row * nb), nb),
            x,
        );
    }
}

/// Q2_K × Q8_K matrix-vector multiplication
/// Uses std::thread::scope for low-overhead parallel execution
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q2_k_q8_k(
    weights: &[BlockQ2K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    use crate::cpu::{SyncPtr, default_n_threads, parallel_chunks};

    let nb = k / QK_K;
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
    let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ2K);
    let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8K);

    parallel_chunks(m, default_n_threads(), move |start, end| {
        for row in start..end {
            *dst_ptr.get().add(row) = vec_dot_q2_k_q8_k_neon(
                k,
                std::slice::from_raw_parts(weights_ptr.get().add(row * nb), nb),
                std::slice::from_raw_parts(x_ptr.get(), nb),
            );
        }
    });
}

/// Single-thread Q2_K × Q8_K matrix-vector multiplication
/// For decode (batch=1) where thread overhead dominates
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q2_k_q8_k_single_thread(
    weights: &[BlockQ2K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    for row in 0..m {
        dst[row] = vec_dot_q2_k_q8_k_neon(
            k,
            std::slice::from_raw_parts(weights.as_ptr().add(row * nb), nb),
            x,
        );
    }
}

/// Q4_0 × Q8_0 matrix-vector multiplication
/// Uses std::thread::scope for low-overhead parallel execution
pub unsafe fn matmul_q4_0_q8_0(
    weights: &[BlockQ4_0],
    x: &[BlockQ8_0],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    use crate::cpu::{SyncPtr, default_n_threads, parallel_chunks};

    let nb = k / QK4_0;
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
    let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ4_0);
    let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8_0);

    parallel_chunks(m, default_n_threads(), move |start, end| {
        for i in start..end {
            *dst_ptr.get().add(i) = vec_dot_q4_0_q8_0_neon(
                k,
                std::slice::from_raw_parts(weights_ptr.get().add(i * nb), nb),
                std::slice::from_raw_parts(x_ptr.get(), nb),
            );
        }
    });
}

/// Single-thread Q4_0 × Q8_0 matrix-vector multiplication
/// For decode (batch=1) where thread overhead dominates
pub unsafe fn matmul_q4_0_q8_0_single_thread(
    weights: &[BlockQ4_0],
    x: &[BlockQ8_0],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK4_0;

    for i in 0..m {
        dst[i] = vec_dot_q4_0_q8_0_neon(
            k,
            std::slice::from_raw_parts(weights.as_ptr().add(i * nb), nb),
            x,
        );
    }
}

/// Q5_0 × Q8_0 matrix-vector multiplication
/// Uses std::thread::scope for low-overhead parallel execution
pub unsafe fn matmul_q5_0_q8_0(
    weights: &[BlockQ5_0],
    x: &[BlockQ8_0],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    use crate::cpu::{SyncPtr, default_n_threads, parallel_chunks};

    let nb = k / QK8_0;
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
    let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ5_0);
    let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8_0);

    parallel_chunks(m, default_n_threads(), move |start, end| {
        for i in start..end {
            *dst_ptr.get().add(i) = vec_dot_q5_0_q8_0_neon(
                k,
                std::slice::from_raw_parts(weights_ptr.get().add(i * nb), nb),
                std::slice::from_raw_parts(x_ptr.get(), nb),
            );
        }
    });
}

/// Single-thread Q5_0 × Q8_0 matrix-vector multiplication
/// For decode (batch=1) where thread overhead dominates
pub unsafe fn matmul_q5_0_q8_0_single_thread(
    weights: &[BlockQ5_0],
    x: &[BlockQ8_0],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK8_0;

    for i in 0..m {
        dst[i] = vec_dot_q5_0_q8_0_neon(
            k,
            std::slice::from_raw_parts(weights.as_ptr().add(i * nb), nb),
            x,
        );
    }
}

/// Q4_1 × Q8_1 matrix-vector multiplication
/// Uses std::thread::scope for low-overhead parallel execution
pub unsafe fn matmul_q4_1_q8_1(
    weights: &[BlockQ4_1],
    x: &[BlockQ8_1],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    use crate::cpu::{SyncPtr, default_n_threads, parallel_chunks};

    let nb = k / QK4_0;
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
    let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ4_1);
    let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8_1);

    parallel_chunks(m, default_n_threads(), move |start, end| {
        for i in start..end {
            *dst_ptr.get().add(i) = vec_dot_q4_1_q8_1_neon(
                k,
                std::slice::from_raw_parts(weights_ptr.get().add(i * nb), nb),
                std::slice::from_raw_parts(x_ptr.get(), nb),
            );
        }
    });
}

/// Single-thread Q4_1 × Q8_1 matrix-vector multiplication
/// For decode (batch=1) where thread overhead dominates
pub unsafe fn matmul_q4_1_q8_1_single_thread(
    weights: &[BlockQ4_1],
    x: &[BlockQ8_1],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK4_0;

    for i in 0..m {
        dst[i] = vec_dot_q4_1_q8_1_neon(
            k,
            std::slice::from_raw_parts(weights.as_ptr().add(i * nb), nb),
            x,
        );
    }
}

/// Q5_1 × Q8_1 matrix-vector multiplication
/// Uses std::thread::scope for low-overhead parallel execution
pub unsafe fn matmul_q5_1_q8_1(
    weights: &[BlockQ5_1],
    x: &[BlockQ8_1],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    use crate::cpu::{SyncPtr, default_n_threads, parallel_chunks};

    let nb = k / QK8_0;
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());
    let weights_ptr = SyncPtr::new(weights.as_ptr() as *mut BlockQ5_1);
    let x_ptr = SyncPtr::new(x.as_ptr() as *mut BlockQ8_1);

    parallel_chunks(m, default_n_threads(), move |start, end| {
        for i in start..end {
            *dst_ptr.get().add(i) = vec_dot_q5_1_q8_1_neon(
                k,
                std::slice::from_raw_parts(weights_ptr.get().add(i * nb), nb),
                std::slice::from_raw_parts(x_ptr.get(), nb),
            );
        }
    });
}

/// Single-thread Q5_1 × Q8_1 matrix-vector multiplication
/// For decode (batch=1) where thread overhead dominates
pub unsafe fn matmul_q5_1_q8_1_single_thread(
    weights: &[BlockQ5_1],
    x: &[BlockQ8_1],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK8_0;

    for i in 0..m {
        dst[i] = vec_dot_q5_1_q8_1_neon(
            k,
            std::slice::from_raw_parts(weights.as_ptr().add(i * nb), nb),
            x,
        );
    }
}

// ===== IQ series matmul =====

/// IQ4_XS × Q8_K matrix-vector multiplication
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq4_xs_q8_k(
    weights: &[BlockIQ4XS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;
    use crate::cpu::{SyncPtr, gemv_work_stealing};
    let n_threads = crate::cpu::default_n_threads();
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());

    gemv_work_stealing(m, n_threads, |i| {
        *dst_ptr.get().add(i) = vec_dot_iq4_xs_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

/// IQ4_NL × Q8_0 matrix-vector multiplication
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq4_nl_q8_0(
    weights: &[BlockIQ4NL],
    x: &[crate::quant::types::BlockQ8_0],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK4_0;
    use crate::cpu::{SyncPtr, gemv_work_stealing};
    let n_threads = crate::cpu::default_n_threads();
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());

    gemv_work_stealing(m, n_threads, |i| {
        *dst_ptr.get().add(i) = vec_dot_iq4_nl_q8_0_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

/// IQ3_XXS × Q8_K matrix-vector multiplication
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq3_xxs_q8_k(
    weights: &[BlockIQ3XXS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;
    use crate::cpu::{SyncPtr, gemv_work_stealing};
    let n_threads = crate::cpu::default_n_threads();
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());

    gemv_work_stealing(m, n_threads, |i| {
        *dst_ptr.get().add(i) = vec_dot_iq3_xxs_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

/// IQ3_S × Q8_K matrix-vector multiplication
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq3_s_q8_k(
    weights: &[BlockIQ3S],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;
    use crate::cpu::{SyncPtr, gemv_work_stealing};
    let n_threads = crate::cpu::default_n_threads();
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());

    gemv_work_stealing(m, n_threads, |i| {
        *dst_ptr.get().add(i) = vec_dot_iq3_s_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

/// IQ2_XXS × Q8_K matrix-vector multiplication
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq2_xxs_q8_k(
    weights: &[BlockIQ2XXS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;
    use crate::cpu::{SyncPtr, gemv_work_stealing};
    let n_threads = crate::cpu::default_n_threads();
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());

    gemv_work_stealing(m, n_threads, |i| {
        *dst_ptr.get().add(i) = vec_dot_iq2_xxs_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

/// IQ2_XS × Q8_K matrix-vector multiplication
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq2_xs_q8_k(
    weights: &[BlockIQ2XS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;
    use crate::cpu::{SyncPtr, gemv_work_stealing};
    let n_threads = crate::cpu::default_n_threads();
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());

    gemv_work_stealing(m, n_threads, |i| {
        *dst_ptr.get().add(i) = vec_dot_iq2_xs_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

/// IQ2_S × Q8_K matrix-vector multiplication
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq2_s_q8_k(
    weights: &[BlockIQ2S],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;
    use crate::cpu::{SyncPtr, gemv_work_stealing};
    let n_threads = crate::cpu::default_n_threads();
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());

    gemv_work_stealing(m, n_threads, |i| {
        *dst_ptr.get().add(i) = vec_dot_iq2_s_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

/// IQ1_S × Q8_K matrix-vector multiplication
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq1_s_q8_k(
    weights: &[BlockIQ1S],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;
    use crate::cpu::{SyncPtr, gemv_work_stealing};
    let n_threads = crate::cpu::default_n_threads();
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());

    gemv_work_stealing(m, n_threads, |i| {
        *dst_ptr.get().add(i) = vec_dot_iq1_s_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

/// IQ1_M × Q8_K matrix-vector multiplication
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq1_m_q8_k(
    weights: &[BlockIQ1M],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;
    use crate::cpu::{SyncPtr, gemv_work_stealing};
    let n_threads = crate::cpu::default_n_threads();
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());

    gemv_work_stealing(m, n_threads, |i| {
        *dst_ptr.get().add(i) = vec_dot_iq1_m_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

// ===== TQ series matmul =====

/// TQ2_0 × Q8_K matrix-vector multiplication
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_tq2_0_q8_k(
    weights: &[BlockTQ2_0],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;
    use crate::cpu::{SyncPtr, gemv_work_stealing};
    let n_threads = crate::cpu::default_n_threads();
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());

    gemv_work_stealing(m, n_threads, |i| {
        *dst_ptr.get().add(i) = vec_dot_tq2_0_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

/// TQ1_0 × Q8_K matrix-vector multiplication
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_tq1_0_q8_k(
    weights: &[BlockTQ1_0],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;
    use crate::cpu::{SyncPtr, gemv_work_stealing};
    let n_threads = crate::cpu::default_n_threads();
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());

    gemv_work_stealing(m, n_threads, |i| {
        *dst_ptr.get().add(i) = vec_dot_tq1_0_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

// ===== F16 matmul =====

/// F16 matrix-vector multiplication
///
/// Optimized GEMV for F16 weight matrices.
/// - weights: F16 data as u16 slice, shape [m, k]
/// - x: F32 input vector, length k (converted to F16 internally)
/// - dst: F32 output vector, length m
/// - k: input dimension
/// - m: output dimension
#[target_feature(enable = "neon")]
pub unsafe fn matmul_f16_f32(
    weights: &[u16], // F16 data
    x: &[f32],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    use crate::ops::arm::neon::dup_f32_to_f16_neon;
    use crate::ops::attention::vec_dot_f16_neon;
    use half::f16;

    // Convert input x to F16 once (avoids repeated conversion)
    let mut x_f16 = vec![0u16; k];
    let x_f16_slice =
        std::slice::from_raw_parts_mut(x_f16.as_mut_ptr() as *mut f16, k);
    dup_f32_to_f16_neon(x, x_f16_slice);

    use crate::cpu::{SyncPtr, gemv_work_stealing};
    let n_threads = crate::cpu::default_n_threads();
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());

    gemv_work_stealing(m, n_threads, |i| {
        *dst_ptr.get().add(i) = vec_dot_f16_neon(
            weights.get_unchecked(i * k..(i + 1) * k),
            &x_f16,
            k,
        );
    });
}

/// Single-thread F16 matrix-vector multiplication
/// For decode (batch=1) where thread overhead dominates
#[target_feature(enable = "neon")]
pub unsafe fn matmul_f16_f32_single_thread(
    weights: &[u16], // F16 data
    x: &[f32],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    use crate::ops::arm::neon::dup_f32_to_f16_neon;
    use crate::ops::attention::vec_dot_f16_neon;
    use half::f16;

    // Convert input x to F16 once
    let mut x_f16 = vec![0u16; k];
    let x_f16_slice =
        std::slice::from_raw_parts_mut(x_f16.as_mut_ptr() as *mut f16, k);
    dup_f32_to_f16_neon(x, x_f16_slice);

    for i in 0..m {
        dst[i] = vec_dot_f16_neon(
            weights.get_unchecked(i * k..(i + 1) * k),
            &x_f16,
            k,
        );
    }
}
