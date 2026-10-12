// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark vec_dot performance comparison
// FP16 vs FP32 accumulation

use std::arch::aarch64::*;
use std::arch::asm;
use std::time::Instant;

const HEAD_DIM: usize = 128;
const NUM_ITER: usize = 10000;

/// FP16 accumulation with inline asm (current approach)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn vec_dot_f16_accum(q: &[u16], k: &[u16], n: usize) -> f32 {
    let mut q_ptr = q.as_ptr();
    let mut k_ptr = k.as_ptr();
    let np = n & !31;

    q_ptr = q_ptr.add(16);
    k_ptr = k_ptr.add(16);

    let mut sum0: float16x8_t;
    asm!(
        "movi v0.2d, #0", "movi v1.2d, #0", "movi v2.2d, #0", "movi v3.2d, #0",
        "mov x8, #0", "cmp x2, #32", "b.lt 2f",
        "1:",
        "ldp q4, q5, [x0, #-32]", "ldp q6, q7, [x1, #-32]",
        "fmla v3.8h, v6.8h, v4.8h", "fmla v2.8h, v7.8h, v5.8h",
        "ldp q4, q5, [x0], #64", "ldp q6, q7, [x1], #64",
        "fmla v1.8h, v6.8h, v4.8h", "fmla v0.8h, v7.8h, v5.8h",
        "add x8, x8, #32", "cmp x8, x2", "b.lo 1b",
        "2:", "fadd v2.8h, v2.8h, v3.8h", "fadd v0.8h, v0.8h, v1.8h", "fadd v0.8h, v0.8h, v2.8h",
        lateout("v0") sum0, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _, out("x8") _,
        inout("x0") q_ptr => _, inout("x1") k_ptr => _, in("x2") np,
        options(nostack)
    );

    let sum_f32_0 = vcvt_f32_f16(vget_low_f16(sum0));
    let sum_f32_1 = vcvt_high_f32_f16(sum0);
    let mut result = vaddvq_f32(vaddq_f32(sum_f32_0, sum_f32_1));

    for j in np..n {
        let q_val = half::f16::from_bits(q[j]).to_f32();
        let k_val = half::f16::from_bits(k[j]).to_f32();
        result += q_val * k_val;
    }
    result
}

/// FP32 accumulation - FP16 input converted to FP32 immediately
/// Uses stable FP32 NEON intrinsics
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn vec_dot_f32_accum(q: &[u16], k: &[u16], n: usize) -> f32 {
    let q_ptr = q.as_ptr();
    let k_ptr = k.as_ptr();
    let np = n & !31;

    // 4-way unroll FP32 accumulators
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    let mut i = 0;
    while i < np {
        // Load FP16 and convert to FP32 (4 values at a time)
        let q0 = vcvt_f32_f16(vld1_f16(q_ptr.add(i) as *const f16));
        let q1 = vcvt_f32_f16(vld1_f16(q_ptr.add(i + 4) as *const f16));
        let q2 = vcvt_f32_f16(vld1_f16(q_ptr.add(i + 8) as *const f16));
        let q3 = vcvt_f32_f16(vld1_f16(q_ptr.add(i + 12) as *const f16));

        let k0 = vcvt_f32_f16(vld1_f16(k_ptr.add(i) as *const f16));
        let k1 = vcvt_f32_f16(vld1_f16(k_ptr.add(i + 4) as *const f16));
        let k2 = vcvt_f32_f16(vld1_f16(k_ptr.add(i + 8) as *const f16));
        let k3 = vcvt_f32_f16(vld1_f16(k_ptr.add(i + 12) as *const f16));

        // FP32 FMA
        sum0 = vfmaq_f32(sum0, q0, k0);
        sum1 = vfmaq_f32(sum1, q1, k1);
        sum2 = vfmaq_f32(sum2, q2, k2);
        sum3 = vfmaq_f32(sum3, q3, k3);

        i += 16;
    }

    // Reduce
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);

    let mut result = vaddvq_f32(sum0);

    for j in np..n {
        let q_val = half::f16::from_bits(q[j]).to_f32();
        let k_val = half::f16::from_bits(k[j]).to_f32();
        result += q_val * k_val;
    }
    result
}

#[cfg(target_arch = "aarch64")]
fn main() {
    let q: Vec<u16> = vec![0x3C00; HEAD_DIM]; // FP16 1.0
    let k: Vec<u16> = vec![0x3800; HEAD_DIM]; // FP16 0.5

    unsafe {
        // Warmup
        for _ in 0..100 {
            let _ = vec_dot_f16_accum(&q, &k, HEAD_DIM);
            let _ = vec_dot_f32_accum(&q, &k, HEAD_DIM);
        }

        // FP16 accumulation benchmark
        let mut sum = 0.0f32;
        let start = Instant::now();
        for _ in 0..NUM_ITER {
            sum += vec_dot_f16_accum(&q, &k, HEAD_DIM);
        }
        let elapsed = start.elapsed();
        let flops = 2.0 * HEAD_DIM as f64 * NUM_ITER as f64;
        let gflops = flops / elapsed.as_secs_f64() / 1e9;
        println!("FP16 accumulation: {:.2} GFLOPS", gflops);

        // FP32 accumulation benchmark
        let start = Instant::now();
        for _ in 0..NUM_ITER {
            sum += vec_dot_f32_accum(&q, &k, HEAD_DIM);
        }
        let elapsed = start.elapsed();
        let gflops = flops / elapsed.as_secs_f64() / 1e9;
        println!("FP32 accumulation: {:.2} GFLOPS", gflops);

        println!("Sum: {} (prevent optimization)", sum);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    println!("NEON only available on aarch64");
}
