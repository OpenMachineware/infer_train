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
/// Uses work-stealing parallel execution (no Rayon overhead)
#[target_feature(enable = "neon,dotprod")]
unsafe fn matmul_q4_k_q8_k_standard(
    weights: &[BlockQ4K],
    x: &[BlockQ8K],
    dst: &mut [f32],
    k: usize,
    m: usize,
    nb: usize,
) {
    use crate::cpu::{SyncPtr, gemv_work_stealing};

    let n_threads = crate::cpu::default_n_threads();
    let dst_ptr = SyncPtr::new(dst.as_mut_ptr());

    gemv_work_stealing(m, n_threads, |i| {
        *dst_ptr.get().add(i) = vec_dot_q4_k_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

/// Q6_K × Q8_K matrix-vector multiplication
/// Uses work-stealing parallel execution (no Rayon overhead)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q6_k_q8_k(
    weights: &[BlockQ6K],
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
        *dst_ptr.get().add(i) = vec_dot_q6_k_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

/// Q5_K × Q8_K matrix-vector multiplication
/// Uses work-stealing parallel execution (no Rayon overhead)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q5_k_q8_k(
    weights: &[BlockQ5K],
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
        *dst_ptr.get().add(i) = vec_dot_q5_k_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

/// Q3_K × Q8_K matrix-vector multiplication
/// Uses work-stealing parallel execution (no Rayon overhead)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q3_k_q8_k(
    weights: &[BlockQ3K],
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
        *dst_ptr.get().add(i) = vec_dot_q3_k_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
}

/// Q2_K × Q8_K matrix-vector multiplication
/// Uses work-stealing parallel execution (no Rayon overhead)
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn matmul_q2_k_q8_k(
    weights: &[BlockQ2K],
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
        *dst_ptr.get().add(i) = vec_dot_q2_k_q8_k_neon(
            k,
            weights.get_unchecked(i * nb..(i + 1) * nb),
            x,
        );
    });
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
