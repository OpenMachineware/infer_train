// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Attention operator NEON implementations
// FP16 inline assembly for stable Rust compatibility

#![allow(asm_sub_register)]

use std::arch::aarch64::*;
use std::arch::asm;

// ============================================================================
// FP16 NEON via inline assembly (stable Rust compatible)
// ============================================================================

/// FP16 FMA using inline assembly: a = a + b * c
/// Uses uint16x8_t (stable) to avoid unstable float16x8_t type
#[inline(always)]
pub unsafe fn vfmaq_f16_asm(
    a: uint16x8_t,
    b: uint16x8_t,
    c: uint16x8_t,
) -> uint16x8_t {
    let result: uint16x8_t;
    asm!(
        "fmla {0}.8h, {1}.8h, {2}.8h",
        inout(vreg) a => result,
        in(vreg) b,
        in(vreg) c,
        options(pure, nomem, nostack)
    );
    result
}

/// FP16 add using inline assembly
/// Uses uint16x8_t (stable) to avoid unstable float16x8_t type
#[inline(always)]
pub unsafe fn vaddq_f16_asm(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    let result: uint16x8_t;
    asm!(
        "fadd {0}.8h, {1}.8h, {2}.8h",
        lateout(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, nostack)
    );
    result
}

/// FP16 mul using inline assembly
/// Uses uint16x8_t (stable) to avoid unstable float16x8_t type
#[inline(always)]
pub unsafe fn vmulq_f16_asm(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    let result: uint16x8_t;
    asm!(
        "fmul {0}.8h, {1}.8h, {2}.8h",
        lateout(vreg) result,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, nostack)
    );
    result
}

/// Convert FP16 vector (as uint16x8_t) to two FP32 vectors using inline asm
/// Returns (low 4 floats, high 4 floats)
#[inline(always)]
pub unsafe fn vcvt_f32_from_f16(v: uint16x8_t) -> (float32x4_t, float32x4_t) {
    let low: float32x4_t;
    let high: float32x4_t;
    asm!(
        "fcvtl {0}.4s, {2}.4h",
        "fcvtl2 {1}.4s, {2}.8h",
        lateout(vreg) low,
        lateout(vreg) high,
        in(vreg) v,
        options(pure, nomem, nostack)
    );
    (low, high)
}

/// Horizontal sum of float32x4_t
#[inline(always)]
pub unsafe fn vaddvq_f32_inline(v: float32x4_t) -> f32 {
    vaddvq_f32(v)
}

/// Convert two FP32 vectors to FP16 vector using inline asm
/// Returns uint16x8_t (FP16 bits) to avoid unstable float16x8_t type
#[inline(always)]
pub unsafe fn vcvt_f16_from_f32(
    low: float32x4_t,
    high: float32x4_t,
) -> uint16x8_t {
    let result: uint16x8_t;
    asm!(
        "fcvtn {0}.4h, {1}.4s",
        "fcvtn2 {0}.8h, {2}.4s",
        lateout(vreg) result,
        in(vreg) low,
        in(vreg) high,
        options(pure, nomem, nostack)
    );
    result
}

/// Load 4 FP16 values and convert to FP32 using inline asm (stable Rust)
#[inline(always)]
pub unsafe fn vld1_f16_and_cvt_to_f32(ptr: *const u16) -> float32x4_t {
    let result: float32x4_t;
    asm!(
        "ld1 {0}.4h, [{1}]",
        "fcvtl {0}.4s, {0}.4h",
        lateout(vreg) result,
        in(reg) ptr,
        options(pure, nomem, nostack)
    );
    result
}

// ============================================================================
// Vec Dot NEON
// ============================================================================

/// Vec dot for FP32 vectors
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut sum = vdupq_n_f32(0.0);

    let chunks = n / 4;
    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        sum = vfmaq_f32(sum, va, vb);
    }

    let mut result = vaddvq_f32(sum);

    // Handle remainder
    for i in (chunks * 4)..n {
        result += a[i] * b[i];
    }

    result
}

/// Load two FP16 pairs for vec_dot
#[inline(always)]
unsafe fn load_pair_f16(ptr: *const u16) -> (uint16x8_t, uint16x8_t) {
    let a = vld1q_u16(ptr);
    let b = vld1q_u16(ptr.add(8));
    (a, b)
}

/// Vec dot for FP16 vectors (stored as u16 bits)
#[target_feature(enable = "neon")]
pub unsafe fn vec_dot_f16(q: &[u16], k: &[u16], n: usize) -> f32 {
    let mut sum = vdupq_n_f32(0.0);
    let chunks = n / 8;

    for i in 0..chunks {
        let (q_low, q_high) = load_pair_f16(q.as_ptr().add(i * 16));
        let (k_low, k_high) = load_pair_f16(k.as_ptr().add(i * 16));

        // Convert to FP32
        let (q0, q1) = vcvt_f32_from_f16(q_low);
        let (q2, q3) = vcvt_f32_from_f16(q_high);
        let (k0, k1) = vcvt_f32_from_f16(k_low);
        let (k2, k3) = vcvt_f32_from_f16(k_high);

        // FMA
        sum = vfmaq_f32(sum, q0, k0);
        sum = vfmaq_f32(sum, q1, k1);
        sum = vfmaq_f32(sum, q2, k2);
        sum = vfmaq_f32(sum, q3, k3);
    }

    vaddvq_f32(sum)
}

/// Convert FP32 to FP16 bits using hardware fcvt instruction
#[inline(always)]
pub unsafe fn fp32_to_fp16_bits(v: f32) -> u16 {
    let result: u16;
    asm!(
        "fcvt h16, s0",
        "fmov {0}, h16",
        out(reg) result,
        in("s0") v,
        out("h16") _,
        options(pure, nomem, nostack)
    );
    result
}

// ============================================================================
// Vec MAD NEON
// ============================================================================

/// Vec MAD for FP16: y += x * v (FP16 storage, FP32 arithmetic)
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f16(y: &mut [u16], x: &[u16], v: f32, n: usize) {
    let v_vec = vdupq_n_f32(v);

    for i in (0..n).step_by(8) {
        let y_ptr = y.as_mut_ptr().add(i);
        let x_ptr = x.as_ptr().add(i);

        // Load x as FP16
        let x_f16 = vld1q_u16(x_ptr);

        // Convert to FP32
        let (x_lo, x_hi) = vcvt_f32_from_f16(x_f16);

        // Compute x * v
        let r_lo = vmulq_f32(x_lo, v_vec);
        let r_hi = vmulq_f32(x_hi, v_vec);

        // Load y
        let y_f16 = vld1q_u16(y_ptr);
        let (y_lo, y_hi) = vcvt_f32_from_f16(y_f16);

        // Add
        let out_lo = vaddq_f32(y_lo, r_lo);
        let out_hi = vaddq_f32(y_hi, r_hi);

        // Convert back to FP16 and store
        let out = vcvt_f16_from_f32(out_lo, out_hi);
        vst1q_u16(y_ptr, out);
    }
}

/// Vec scale for FP16
#[target_feature(enable = "neon")]
pub unsafe fn vec_scale_f16(y: &mut [u16], v: f32, n: usize) {
    let v_vec = vdupq_n_f32(v);

    for i in (0..n).step_by(8) {
        let y_ptr = y.as_mut_ptr().add(i);

        // Load y as FP16
        let y_f16 = vld1q_u16(y_ptr);

        // Convert to FP32
        let (y_lo, y_hi) = vcvt_f32_from_f16(y_f16);

        // Scale
        let r_lo = vmulq_f32(y_lo, v_vec);
        let r_hi = vmulq_f32(y_hi, v_vec);

        // Convert back to FP16 and store
        let out = vcvt_f16_from_f32(r_lo, r_hi);
        vst1q_u16(y_ptr, out);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert FP16 bits to FP32
#[inline(always)]
pub unsafe fn fp16_to_fp32(bits: u16) -> f32 {
    let f16 = vreinterpret_f16_u16(vdup_n_u16(bits));
    let f32_vec = vcvt_f32_f16(f16);
    vgetq_lane_f32(f32_vec, 0)
}

/// Convert BF16 bits to FP32
#[inline(always)]
pub unsafe fn bf16_to_fp32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Dequantize BF16 row to FP32
#[target_feature(enable = "neon")]
pub unsafe fn dequantize_row_bf16(x: &[u16], y: &mut [f32]) {
    let n = x.len();
    let chunks = n / 4;

    for i in 0..chunks {
        let bf16 = vld1_u16(x.as_ptr().add(i * 4));
        // BF16 is just FP32 bits shifted left by 16
        // vcvt would interpret as FP16, so we need manual conversion
        let bf32 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(bf16), 16));
        vst1q_f32(y.as_mut_ptr().add(i * 4), bf32);
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        y[i] = bf16_to_fp32(x[i]);
    }
}

/// Dequantize FP16 row to FP32
#[target_feature(enable = "neon")]
pub unsafe fn dequantize_row_f16(x: &[u16], y: &mut [f32]) {
    let n = x.len();
    let chunks = n / 8;

    for i in 0..chunks {
        let f16 = vld1q_u16(x.as_ptr().add(i * 8));
        let (lo, hi) = vcvt_f32_from_f16(f16);
        vst1q_f32(y.as_mut_ptr().add(i * 8), lo);
        vst1q_f32(y.as_mut_ptr().add(i * 8 + 4), hi);
    }

    // Handle remainder
    for i in (chunks * 8)..n {
        y[i] = fp16_to_fp32(x[i]);
    }
}

/// Vec dot for FP32 query and FP16 KV
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_f32_f16(q: &[f32], k: &[u16], n: usize) -> f32 {
    let mut sum = vdupq_n_f32(0.0);
    let chunks = n / 8;

    for i in 0..chunks {
        let q_vec = vld1q_f32(q.as_ptr().add(i * 8));
        let k_f16 = vld1q_u16(k.as_ptr().add(i * 8));
        let (k_lo, k_hi) = vcvt_f32_from_f16(k_f16);

        sum = vfmaq_f32(sum, q_vec, k_lo);
        sum = vfmaq_f32(sum, vld1q_f32(q.as_ptr().add(i * 8 + 4)), k_hi);
    }

    let mut result = vaddvq_f32(sum);

    // Handle remainder
    for i in (chunks * 8)..n {
        result += q[i] * fp16_to_fp32(k[i]);
    }

    result
}

/// Vec MAD for FP32 output and FP16 V
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f32_f16(y: &mut [f32], v: &[u16], weight: f32, n: usize) {
    let w_vec = vdupq_n_f32(weight);

    for i in (0..n).step_by(8) {
        let v_f16 = vld1q_u16(v.as_ptr().add(i));
        let (v_lo, v_hi) = vcvt_f32_from_f16(v_f16);

        let y_lo = vld1q_f32(y.as_ptr().add(i));
        let y_hi = vld1q_f32(y.as_ptr().add(i + 4));

        let r_lo = vfmaq_f32(y_lo, v_lo, w_vec);
        let r_hi = vfmaq_f32(y_hi, v_hi, w_vec);

        vst1q_f32(y.as_mut_ptr().add(i), r_lo);
        vst1q_f32(y.as_mut_ptr().add(i + 4), r_hi);
    }
}
