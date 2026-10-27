// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#![allow(asm_sub_register)]

use crate::cpu::thread_pool::GLOBAL_THREAD_POOL;
use crate::quant::types::*;
use crate::quant::vec_dot::arm::*;
use once_cell::sync::Lazy;
use std::arch::aarch64::*;
use std::arch::asm;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// GEMV task parameters for persistent thread pool
#[repr(C)]
struct GemvTaskParams {
    weights_ptr: *const u8, // Type-erased weights pointer
    x_ptr: *const u8,       // Type-erased input pointer
    dst_ptr: *mut f32,
    k: usize,
    m: usize,
    nb: usize,
    dtype: u8, // Dtype code
}

unsafe impl Send for GemvTaskParams {}
unsafe impl Sync for GemvTaskParams {}

// Dtype codes for task function
const DTYPE_Q4_K: u8 = 0;
const DTYPE_Q5_K: u8 = 1;
const DTYPE_Q6_K: u8 = 2;
const DTYPE_Q3_K: u8 = 3;
const DTYPE_Q2_K: u8 = 4;
const DTYPE_IQ4_XS: u8 = 5;
const DTYPE_IQ4_NL: u8 = 6;
const DTYPE_IQ3_XXS: u8 = 7;
const DTYPE_IQ3_S: u8 = 8;
const DTYPE_IQ2_XXS: u8 = 9;
const DTYPE_IQ2_XS: u8 = 10;
const DTYPE_IQ2_S: u8 = 11;
const DTYPE_IQ1_S: u8 = 12;
const DTYPE_IQ1_M: u8 = 13;
const DTYPE_TQ2_0: u8 = 14;
const DTYPE_TQ1_0: u8 = 15;
const DTYPE_F16: u8 = 16;

/// GEMV task function for persistent thread pool
/// NOTE: user_data is a leaked Box<GemvTaskParams> pointer. The caller must free it after execute returns.
unsafe fn gemv_task_fn(
    counter: *const AtomicUsize,
    chunk_size: usize,
    n_items: usize,
    user_data: usize,
) {
    // CRITICAL: Copy all params to local variables IMMEDIATELY
    // This eliminates lifetime dependency on the Box
    let params = &*(user_data as *const GemvTaskParams);
    let weights_ptr = params.weights_ptr;
    let x_ptr = params.x_ptr;
    let dst_ptr = params.dst_ptr;
    let k = params.k;
    let m = params.m;
    let nb = params.nb;
    let dtype = params.dtype;

    let counter = &*counter;

    // Safety check: if no work to do, return early
    if m == 0 || k == 0 || (nb == 0 && dtype != DTYPE_F16) {
        return;
    }

    // Ensure we see the counter reset
    std::sync::atomic::fence(Ordering::Acquire);

    match dtype {
        DTYPE_Q4_K => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockQ4K,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_q4_k_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_Q5_K => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockQ5K,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_q5_k_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_Q6_K => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockQ6K,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_q6_k_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_Q3_K => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockQ3K,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_q3_k_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_Q2_K => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockQ2K,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_q2_k_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_IQ4_XS => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockIQ4XS,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_iq4_xs_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_IQ4_NL => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockIQ4NL,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8_0, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_iq4_nl_q8_0_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_IQ3_XXS => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockIQ3XXS,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_iq3_xxs_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_IQ3_S => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockIQ3S,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_iq3_s_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_IQ2_XXS => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockIQ2XXS,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_iq2_xxs_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_IQ2_XS => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockIQ2XS,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_iq2_xs_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_IQ2_S => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockIQ2S,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_iq2_s_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_IQ1_S => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockIQ1S,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_iq1_s_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_IQ1_M => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockIQ1M,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_iq1_m_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_TQ2_0 => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockTQ2_0,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_tq2_0_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_TQ1_0 => {
            let weights = std::slice::from_raw_parts(
                weights_ptr as *const BlockTQ1_0,
                m * nb,
            );
            let x = std::slice::from_raw_parts(x_ptr as *const BlockQ8K, nb);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    dst[i] = vec_dot_tq1_0_q8_k_neon(
                        k,
                        &weights[i * nb..(i + 1) * nb],
                        x,
                    );
                }
            }
        }
        DTYPE_F16 => {
            use crate::ops::attention::vec_dot_f16_neon;

            let weights =
                std::slice::from_raw_parts(weights_ptr as *const u16, m * k);
            let x = std::slice::from_raw_parts(x_ptr as *const u16, k);
            let dst = std::slice::from_raw_parts_mut(dst_ptr, m);

            loop {
                let chunk_start =
                    counter.fetch_add(chunk_size, Ordering::Relaxed);
                if chunk_start >= n_items {
                    break;
                }
                let chunk_end = (chunk_start + chunk_size).min(n_items);
                for i in chunk_start..chunk_end {
                    let row_start = i * k;
                    dst[i] = vec_dot_f16_neon(
                        &x,
                        &weights[row_start..row_start + k],
                        k,
                    );
                }
            }
        }
        _ => unreachable!(),
    }
}

/// Cache for interleaved Q4_K weights
/// Key: pointer to original weights
/// Value: interleaved weights (m/8 blocks of BlockQ4Kx8)
static Q4K_INTERLEAVED_CACHE: Lazy<
    Arc<Mutex<HashMap<usize, Vec<BlockQ4Kx8>>>>,
> = Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

/// Manual SDOT implementation via inline asm
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
/// Uses interleaved weight layout when possible for better cache locality
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q4_k_q8_k(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    // Use interleaved kernel if m is divisible by 8 and >= 8
    if m >= 8 && m % 8 == 0 {
        // Get or create interleaved weights from cache
        let key = weights.as_ptr() as usize;
        let cache = Q4K_INTERLEAVED_CACHE.clone();
        let mut cache_guard = cache.lock().unwrap();

        let interleaved = if let Some(cached) = cache_guard.get(&key) {
            cached
        } else {
            // Repack weights to interleaved format
            let interleaved = repack_q4_k_batch(weights, m, nb);
            cache_guard.insert(key, interleaved);
            cache_guard.get(&key).unwrap()
        };

        // Use interleaved GEMV kernel
        gemv_q4_k_interleaved(interleaved, x, dst, k, m);
    } else {
        // Fallback to standard kernel for non-multiple-of-8 rows
        matmul_q4_k_q8_k_standard(weights, x, dst, k, m, nb);
    }
}

/// Standard Q4_K × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
unsafe fn matmul_q4_k_q8_k_standard(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
    nb: usize,
) {
    // Heap-allocate params to ensure they survive until task completes
    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_Q4_K,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    // Free the params after execute completes
    drop(Box::from_raw(params_ptr));
}

/// Q6_K × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q6_k_q8_k(
    weights: &[BlockQ6K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    // Heap-allocate params to ensure they survive until task completes
    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_Q6_K,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    // Free the params after execute completes
    drop(Box::from_raw(params_ptr));
}

/// Q5_K × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q5_k_q8_k(
    weights: &[BlockQ5K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    // Heap-allocate params to ensure they survive until task completes
    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_Q5_K,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    // Free the params after execute completes
    drop(Box::from_raw(params_ptr));
}

/// Q3_K × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q3_k_q8_k(
    weights: &[BlockQ3K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_Q3_K,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}

/// Q2_K × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q2_k_q8_k(
    weights: &[BlockQ2K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_Q2_K,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}

/// Q4_0 × Q8_0 matrix-vector multiplication
pub unsafe fn matmul_q4_0_q8_0(
    weights: &[BlockQ4_0],
    x: &[BlockQ8_0],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK4_0;

    for i in 0..m {
        let row_sum =
            vec_dot_q4_0_q8_0_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// Q5_0 × Q8_0 matrix-vector multiplication
pub unsafe fn matmul_q5_0_q8_0(
    weights: &[BlockQ5_0],
    x: &[BlockQ8_0],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK8_0;

    for i in 0..m {
        let row_sum =
            vec_dot_q5_0_q8_0_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// Q4_1 × Q8_1 matrix-vector multiplication
pub unsafe fn matmul_q4_1_q8_1(
    weights: &[BlockQ4_1],
    x: &[BlockQ8_1],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK4_0;

    for i in 0..m {
        let row_sum =
            vec_dot_q4_1_q8_1_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

/// Q5_1 × Q8_1 matrix-vector multiplication
pub unsafe fn matmul_q5_1_q8_1(
    weights: &[BlockQ5_1],
    x: &[BlockQ8_1],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK8_0;

    for i in 0..m {
        let row_sum =
            vec_dot_q5_1_q8_1_neon(k, &weights[i * nb..(i + 1) * nb], x);
        dst[i] = row_sum;
    }
}

// ===== IQ series matmul =====

/// IQ4_XS × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq4_xs_q8_k(
    weights: &[BlockIQ4XS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_IQ4_XS,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}

/// IQ4_NL × Q8_0 matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq4_nl_q8_0(
    weights: &[BlockIQ4NL],
    x: &[crate::quant::types::BlockQ8_0],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK4_0;

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_IQ4_NL,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}

/// IQ3_XXS × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq3_xxs_q8_k(
    weights: &[BlockIQ3XXS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_IQ3_XXS,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}

/// IQ3_S × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq3_s_q8_k(
    weights: &[BlockIQ3S],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_IQ3_S,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}

/// IQ2_XXS × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq2_xxs_q8_k(
    weights: &[BlockIQ2XXS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_IQ2_XXS,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}

/// IQ2_XS × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq2_xs_q8_k(
    weights: &[BlockIQ2XS],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_IQ2_XS,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}

/// IQ2_S × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq2_s_q8_k(
    weights: &[BlockIQ2S],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_IQ2_S,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}

/// IQ1_S × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq1_s_q8_k(
    weights: &[BlockIQ1S],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_IQ1_S,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}

/// IQ1_M × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_iq1_m_q8_k(
    weights: &[BlockIQ1M],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_IQ1_M,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}

// ===== TQ series matmul =====

/// TQ2_0 × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_tq2_0_q8_k(
    weights: &[BlockTQ2_0],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_TQ2_0,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}

/// TQ1_0 × Q8_K matrix-vector multiplication
/// Uses persistent thread pool with work stealing for zero thread creation overhead
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_tq1_0_q8_k(
    weights: &[BlockTQ1_0],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
) {
    let nb = k / QK_K;

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb,
        dtype: DTYPE_TQ1_0,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}

// ===== F16 matmul =====

/// F16 matrix-vector multiplication
///
/// Optimized GEMV for F16 weight matrices.
/// Uses persistent thread pool with work stealing for zero thread creation overhead.
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
    use half::f16;

    // Convert input x to F16 once (avoids repeated conversion)
    let mut x_f16 = vec![0u16; k];
    let x_f16_slice =
        std::slice::from_raw_parts_mut(x_f16.as_mut_ptr() as *mut f16, k);
    dup_f32_to_f16_neon(x, x_f16_slice);

    let params = Box::new(GemvTaskParams {
        weights_ptr: weights.as_ptr() as *const u8,
        x_ptr: x_f16.as_ptr() as *const u8,
        dst_ptr: dst.as_mut_ptr(),
        k,
        m,
        nb: k, // For F16, nb = k (not divided by block size)
        dtype: DTYPE_F16,
    });

    let params_ptr = Box::into_raw(params);
    GLOBAL_THREAD_POOL.execute(gemv_task_fn, m, 64, params_ptr as usize);
    drop(Box::from_raw(params_ptr));
}
