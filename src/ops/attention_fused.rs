// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Fused vec_dot and vec_mad for KV cache quantization formats
// This enables hardware INT8 dot product acceleration (SDOT instruction)

use crate::quant::kv_cache::{
    BlockIQ4NL, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0,
    BlockQ8_1,
};

// ============================================================================
// Q8_0 Quantization for Query
// ============================================================================

/// Quantize FP32 query to Q8_0 for use in fused vec_dot
#[cfg(target_arch = "aarch64")]
pub fn quantize_query_to_q8_0(q: &[f32], q_q8: &mut [BlockQ8_0]) {
    unsafe { crate::ops::arm::neon::quantize_query_to_q8_0(q, q_q8) };
}

/// Quantize FP32 query to Q8_1 for use in Q4_1/Q5_1 fused vec_dot
#[cfg(target_arch = "aarch64")]
pub fn quantize_query_to_q8_1(q: &[f32], q_q8: &mut [BlockQ8_1]) {
    crate::ops::arm::neon::quantize_query_to_q8_1(q, q_q8);
}

// ============================================================================
// Helper Functions
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn fp16_to_fp32_scalar(bits: u16) -> f32 {
    unsafe { crate::ops::arm::neon::fp16_to_fp32_scalar(bits) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn bf16_to_fp32_scalar(bits: u16) -> f32 {
    unsafe { crate::ops::arm::neon::bf16_to_fp32_scalar(bits) }
}

// Scalar fallback for non-aarch64
#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn fp16_to_fp32_scalar(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
pub fn bf16_to_fp32_scalar(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ============================================================================
// Vec Dot: Q8_0 × Qn_0 (direct transmute, no allocation)
// ============================================================================

/// Q8_0 × Q8_0 fused vec_dot (using types from quant/types)
/// Direct transmute - no allocation
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q8_0_q8_0(n: usize, q: &[BlockQ8_0], k: &[BlockQ8_0]) -> f32 {
    unsafe { crate::ops::arm::neon::vec_dot_q8_0_q8_0(n, q, k) }
}

/// Q4_0 × Q8_0 fused vec_dot
/// Direct transmute - no allocation
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q4_0_q8_0(n: usize, q: &[BlockQ8_0], k: &[BlockQ4_0]) -> f32 {
    unsafe { crate::ops::arm::neon::vec_dot_q4_0_q8_0(n, q, k) }
}

/// Q5_0 × Q8_0 fused vec_dot
/// Direct transmute - no allocation
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q5_0_q8_0(n: usize, q: &[BlockQ8_0], k: &[BlockQ5_0]) -> f32 {
    unsafe { crate::ops::arm::neon::vec_dot_q5_0_q8_0(n, q, k) }
}

/// Q4_1 × Q8_1 fused vec_dot using NEON SDOT
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q4_1_q8_1(n: usize, q: &[BlockQ8_0], k: &[BlockQ4_1]) -> f32 {
    unsafe { crate::ops::arm::neon::vec_dot_q4_1_q8_1(n, q, k) }
}

/// Q5_1 × Q8_1 fused vec_dot (scalar fallback)
/// Note: This expects Q8_1, but we're passing Q8_0
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_q5_1_q8_1(n: usize, q: &[BlockQ8_0], k: &[BlockQ5_1]) -> f32 {
    unsafe { crate::ops::arm::neon::vec_dot_q5_1_q8_1(n, q, k) }
}

/// IQ4_NL × Q8_0 fused vec_dot using existing SIMD implementation
/// Direct transmute to types module blocks (same repr(C))
#[cfg(target_arch = "aarch64")]
pub fn vec_dot_iq4_nl_q8_0(n: usize, q: &[BlockQ8_0], k: &[BlockIQ4NL]) -> f32 {
    unsafe { crate::ops::arm::neon::vec_dot_iq4_nl_q8_0(n, q, k) }
}

// ============================================================================
// Vec MAD for V Accumulation
// These are simpler than vec_dot because we accumulate to FP32
// ============================================================================

/// Fused vec_mad: y += weight * V (Q8_0 format)
/// Optimized: uses NEON for i8->f32 conversion and FMA
#[cfg(target_arch = "aarch64")]
pub fn vec_mad_f32_q8_0(y: &mut [f32], v: &[BlockQ8_0], weight: f32, n: usize) {
    unsafe { crate::ops::arm::neon::vec_mad_f32_q8_0(y, v, weight, n) }
}

/// Fused vec_mad: y += weight * V (Q4_0 format)
/// Optimized: NEON vectorized dequantization
#[cfg(target_arch = "aarch64")]
pub fn vec_mad_f32_q4_0(y: &mut [f32], v: &[BlockQ4_0], weight: f32, n: usize) {
    unsafe { crate::ops::arm::neon::vec_mad_f32_q4_0(y, v, weight, n) }
}

/// Fused vec_mad: y += weight * V (Q4_1 format)
/// Optimized: NEON vectorized dequantization with min offset
#[cfg(target_arch = "aarch64")]
pub fn vec_mad_f32_q4_1(y: &mut [f32], v: &[BlockQ4_1], weight: f32, n: usize) {
    unsafe { crate::ops::arm::neon::vec_mad_f32_q4_1(y, v, weight, n) }
}

/// Fused vec_mad: y += weight * V (Q5_0 format)
#[cfg(target_arch = "aarch64")]
pub fn vec_mad_f32_q5_0(y: &mut [f32], v: &[BlockQ5_0], weight: f32, n: usize) {
    unsafe { crate::ops::arm::neon::vec_mad_f32_q5_0(y, v, weight, n) }
}

/// Fused vec_mad: y += weight * V (Q5_1 format)
#[cfg(target_arch = "aarch64")]
pub fn vec_mad_f32_q5_1(y: &mut [f32], v: &[BlockQ5_1], weight: f32, n: usize) {
    unsafe { crate::ops::arm::neon::vec_mad_f32_q5_1(y, v, weight, n) }
}

/// Fused vec_mad: y += weight * V (IQ4_NL format)
/// Optimized: use KVALUES lookup table for dequantization
#[cfg(target_arch = "aarch64")]
pub fn vec_mad_f32_iq4_nl(
    y: &mut [f32],
    v: &[BlockIQ4NL],
    weight: f32,
    n: usize,
) {
    unsafe { crate::ops::arm::neon::vec_mad_f32_iq4_nl(y, v, weight, n) }
}

// ============================================================================
// Scalar fallbacks for non-aarch64
// ============================================================================

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_mad_f32_q8_0(y: &mut [f32], v: &[BlockQ8_0], weight: f32, n: usize) {
    let nb = n / QK;

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let combined_scale = scale * weight;

        for j in 0..QK {
            let val = block.qs[j] as f32 * combined_scale;
            y[b * QK + j] += val;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_mad_f32_q4_0(y: &mut [f32], v: &[BlockQ4_0], weight: f32, n: usize) {
    let nb = n / QK;

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let ws = weight * scale;

        for j in 0..QK / 2 {
            let packed = block.qs[j];
            let lo = (packed & 0x0F) as i8 as f32;
            let hi = ((packed >> 4) & 0x0F) as i8 as f32;

            y[b * QK + j] += ws * lo;
            y[b * QK + j + QK / 2] += ws * hi;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_mad_f32_q4_1(y: &mut [f32], v: &[BlockQ4_1], weight: f32, n: usize) {
    let nb = n / QK;

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let min = fp16_to_fp32_scalar(block.m);
        let w = weight;

        for j in 0..QK / 2 {
            let packed = block.qs[j];
            let lo = (packed & 0x0F) as f32;
            let hi = ((packed >> 4) & 0x0F) as f32;

            let val_lo = w * (scale * lo + min);
            let val_hi = w * (scale * hi + min);

            y[b * QK + j] += val_lo;
            y[b * QK + j + QK / 2] += val_hi;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_mad_f32_q5_0(y: &mut [f32], v: &[BlockQ5_0], weight: f32, n: usize) {
    let nb = n / QK;

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let ws = weight * scale;
        let qh = u32::from_le_bytes(block.qh);

        for j in 0..QK / 2 {
            let packed = block.qs[j];
            let lo = (packed & 0x0F) as i8 as i32;
            let hi = ((packed >> 4) & 0x0F) as i8 as i32;

            let bit_lo = ((qh >> j) & 1) as i32;
            let bit_hi = ((qh >> (j + 16)) & 1) as i32;

            let val_lo = ws * ((lo + bit_lo * 16 - 16) as f32);
            let val_hi = ws * ((hi + bit_hi * 16 - 16) as f32);

            y[b * QK + j] += val_lo;
            y[b * QK + j + QK / 2] += val_hi;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_mad_f32_q5_1(y: &mut [f32], v: &[BlockQ5_1], weight: f32, n: usize) {
    let nb = n / QK;

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let min = fp16_to_fp32_scalar(block.m);
        let w = weight;
        let qh = u32::from_le_bytes(block.qh);

        for j in 0..QK / 2 {
            let packed = block.qs[j];
            let lo = (packed & 0x0F) as f32;
            let hi = ((packed >> 4) & 0x0F) as f32;

            let bit_lo = ((qh >> j) & 1) as f32;
            let bit_hi = ((qh >> (j + 16)) & 1) as f32;

            let val_lo = w * (scale * (lo + bit_lo * 16.0) + min);
            let val_hi = w * (scale * (hi + bit_hi * 16.0) + min);

            y[b * QK + j] += val_lo;
            y[b * QK + j + QK / 2] += val_hi;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub fn vec_mad_f32_iq4_nl(
    y: &mut [f32],
    v: &[BlockIQ4NL],
    weight: f32,
    n: usize,
) {
    let nb = n / QK;

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let ws = weight * scale;

        for j in 0..QK / 2 {
            let packed = block.qs[j];
            let lo = (packed & 0x0F) as usize;
            let hi = ((packed >> 4) & 0x0F) as usize;

            let val_lo = ws * KVALUES_IQ4NL[lo] as f32;
            let val_hi = ws * KVALUES_IQ4NL[hi] as f32;

            y[b * QK + j] += val_lo;
            y[b * QK + j + QK / 2] += val_hi;
        }
    }
}
