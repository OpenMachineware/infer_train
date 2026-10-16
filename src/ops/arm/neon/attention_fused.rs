// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Fused vec_dot and vec_mad for KV cache quantization formats
// This enables hardware INT8 dot product acceleration (SDOT instruction)

use std::arch::aarch64::*;

use crate::quant::kv_cache::quant::{
    quantize_row_q8_0_neon, quantize_row_q8_1,
};
use crate::quant::kv_cache::{
    BlockIQ4NL, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0,
    BlockQ8_1, QK,
};

// ============================================================================
// Q8_0 Quantization for Query
// ============================================================================

/// Quantize FP32 query to Q8_0 for use in fused vec_dot
#[target_feature(enable = "neon")]
pub unsafe fn quantize_query_to_q8_0(q: &[f32], q_q8: &mut [BlockQ8_0]) {
    quantize_row_q8_0_neon(q, q_q8);
}

/// Quantize FP32 query to Q8_1 for use in Q4_1/Q5_1 fused vec_dot
pub fn quantize_query_to_q8_1(q: &[f32], q_q8: &mut [BlockQ8_1]) {
    quantize_row_q8_1(q, q_q8);
}

// ============================================================================
// Helper Functions
// ============================================================================

#[inline]
pub unsafe fn fp16_to_fp32_scalar(bits: u16) -> f32 {
    let f16 = vreinterpret_f16_u16(vdup_n_u16(bits));
    let f32_vec = vcvt_f32_f16(f16);
    vgetq_lane_f32(f32_vec, 0)
}

#[inline]
pub unsafe fn bf16_to_fp32_scalar(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ============================================================================
// Vec Dot: Q8_0 × Qn_0 (direct transmute, no allocation)
// ============================================================================

/// Q8_0 × Q8_0 fused vec_dot (using types from quant/types)
/// Direct transmute - no allocation
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q8_0_q8_0(
    n: usize,
    q: &[BlockQ8_0],
    k: &[BlockQ8_0],
) -> f32 {
    // Transmute to types module blocks (same repr(C))
    let q_types = std::slice::from_raw_parts(
        q.as_ptr() as *const crate::quant::types::BlockQ8_0,
        q.len(),
    );
    let k_types = std::slice::from_raw_parts(
        k.as_ptr() as *const crate::quant::types::BlockQ8_0,
        k.len(),
    );
    crate::quant::vec_dot::arm::q_0_1::vec_dot_q8_0_q8_0_neon(
        n, q_types, k_types,
    )
}

/// Q4_0 × Q8_0 fused vec_dot
/// Direct transmute - no allocation
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q4_0_q8_0(
    n: usize,
    q: &[BlockQ8_0],
    k: &[BlockQ4_0],
) -> f32 {
    let q_types = std::slice::from_raw_parts(
        q.as_ptr() as *const crate::quant::types::BlockQ8_0,
        q.len(),
    );
    let k_types = std::slice::from_raw_parts(
        k.as_ptr() as *const crate::quant::types::BlockQ4_0,
        k.len(),
    );
    crate::quant::vec_dot::arm::q_0_1::vec_dot_q4_0_q8_0_neon(
        n, k_types, q_types,
    )
}

/// Q5_0 × Q8_0 fused vec_dot
/// Direct transmute - no allocation
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q5_0_q8_0(
    n: usize,
    q: &[BlockQ8_0],
    k: &[BlockQ5_0],
) -> f32 {
    let q_types = std::slice::from_raw_parts(
        q.as_ptr() as *const crate::quant::types::BlockQ8_0,
        q.len(),
    );
    let k_types = std::slice::from_raw_parts(
        k.as_ptr() as *const crate::quant::types::BlockQ5_0,
        k.len(),
    );
    crate::quant::vec_dot::arm::q_0_1::vec_dot_q5_0_q8_0_neon(
        n, k_types, q_types,
    )
}

/// Q4_1 × Q8_1 fused vec_dot using NEON SDOT
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q4_1_q8_1(
    n: usize,
    q: &[BlockQ8_0],
    k: &[BlockQ4_1],
) -> f32 {
    // Note: This expects Q8_1, but we're passing Q8_0
    // Q8_1 has an additional 's' field that Q8_0 doesn't have
    // For now, use the scalar fallback since we don't have proper Q8_1 quantization
    // TODO: Implement proper Q8_1 quantization for queries
    scalar_vec_dot_q4_1_q8_0(n, q, k)
}

/// Scalar fallback for Q4_1 × Q8_0
unsafe fn scalar_vec_dot_q4_1_q8_0(
    n: usize,
    q: &[BlockQ8_0],
    k: &[BlockQ4_1],
) -> f32 {
    let nb = n / QK;
    let mut sum = 0.0f32;

    for b in 0..nb {
        let k_block = &k[b];
        let q_block = &q[b];

        let scale_k = fp16_to_fp32_scalar(k_block.d);
        let min_k = fp16_to_fp32_scalar(k_block.m);
        let scale_q = fp16_to_fp32_scalar(q_block.d);

        for j in 0..QK / 2 {
            let packed = k_block.qs[j];
            let lo = (packed & 0x0F) as f32;
            let hi = ((packed >> 4) & 0x0F) as f32;

            let k_lo = scale_k * lo + min_k;
            let k_hi = scale_k * hi + min_k;

            let q_lo = q_block.qs[j] as f32 * scale_q;
            let q_hi = q_block.qs[j + QK / 2] as f32 * scale_q;

            sum += q_lo * k_lo + q_hi * k_hi;
        }
    }
    sum
}

/// Q5_1 × Q8_1 fused vec_dot (scalar fallback)
/// Note: This expects Q8_1, but we're passing Q8_0
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q5_1_q8_1(
    n: usize,
    q: &[BlockQ8_0],
    k: &[BlockQ5_1],
) -> f32 {
    let nb = n / QK;
    let mut sum = 0.0f32;

    for b in 0..nb {
        let k_block = &k[b];
        let q_block = &q[b];

        let scale_k = fp16_to_fp32_scalar(k_block.d);
        let min_k = fp16_to_fp32_scalar(k_block.m);
        let scale_q = fp16_to_fp32_scalar(q_block.d);
        let qh = u32::from_le_bytes(k_block.qh);

        // Process with SIMD where possible
        let mut block_sum = 0.0f32;
        for j in 0..QK / 2 {
            let packed = k_block.qs[j];
            let lo = (packed & 0x0F) as f32;
            let hi = ((packed >> 4) & 0x0F) as f32;

            let bit_lo = ((qh >> j) & 1) as f32;
            let bit_hi = ((qh >> (j + 16)) & 1) as f32;

            let k_lo = scale_k * (lo + bit_lo * 16.0) + min_k;
            let k_hi = scale_k * (hi + bit_hi * 16.0) + min_k;

            let q_lo = q_block.qs[j] as f32 * scale_q;
            let q_hi = q_block.qs[j + QK / 2] as f32 * scale_q;

            block_sum += q_lo * k_lo + q_hi * k_hi;
        }
        sum += block_sum;
    }

    sum
}

/// IQ4_NL × Q8_0 fused vec_dot using existing SIMD implementation
/// Direct transmute to types module blocks (same repr(C))
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_iq4_nl_q8_0(
    n: usize,
    q: &[BlockQ8_0],
    k: &[BlockIQ4NL],
) -> f32 {
    let q_types = std::slice::from_raw_parts(
        q.as_ptr() as *const crate::quant::types::BlockQ8_0,
        q.len(),
    );
    let k_types = std::slice::from_raw_parts(
        k.as_ptr() as *const crate::quant::types::BlockIQ4NL,
        k.len(),
    );
    crate::quant::vec_dot::arm::iq::vec_dot_iq4_nl_q8_0_neon(
        n, k_types, q_types,
    )
}

// ============================================================================
// Vec MAD for V Accumulation
// These are simpler than vec_dot because we accumulate to FP32
// ============================================================================

/// Fused vec_mad: y += weight * V (Q8_0 format)
/// Optimized: uses NEON for i8->f32 conversion and FMA
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f32_q8_0(
    y: &mut [f32],
    v: &[BlockQ8_0],
    weight: f32,
    n: usize,
) {
    let nb = n / QK;

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let combined_scale = scale * weight;

        let y_ptr = y.as_mut_ptr().add(b * QK);
        let w_vec = vdupq_n_f32(combined_scale);

        for i in (0..QK).step_by(8) {
            // Load 8 i8 values
            let k_i8 = vld1_s8(block.qs.as_ptr().add(i));
            // Widen to 8 i16
            let k_i16 = vmovl_s8(k_i8);
            // Split and widen to 8 i32
            let k_i32_0 = vmovl_s16(vget_low_s16(k_i16));
            let k_i32_1 = vmovl_s16(vget_high_s16(k_i16));
            // Convert to f32
            let k_f32_0 = vcvtq_f32_s32(k_i32_0);
            let k_f32_1 = vcvtq_f32_s32(k_i32_1);

            // Load y and do FMA: y += weight * scale * k
            let y_0 = vld1q_f32(y_ptr.add(i));
            let y_1 = vld1q_f32(y_ptr.add(i + 4));

            // Use FMA for better precision and performance
            vst1q_f32(y_ptr.add(i), vfmaq_f32(y_0, k_f32_0, w_vec));
            vst1q_f32(y_ptr.add(i + 4), vfmaq_f32(y_1, k_f32_1, w_vec));
        }
    }
}

/// Fused vec_mad: y += weight * V (Q4_0 format)
/// Optimized: NEON vectorized dequantization
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f32_q4_0(
    y: &mut [f32],
    v: &[BlockQ4_0],
    weight: f32,
    n: usize,
) {
    let nb = n / QK;
    let w_vec = vdupq_n_f32(weight);
    let m4b = vdupq_n_u8(0x0F);
    let s8b = vdupq_n_s8(0x8);

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let scale_vec = vdupq_n_f32(scale);
        let y_ptr = y.as_mut_ptr().add(b * QK);

        // Load 16 bytes of packed 4-bit values
        let packed = vld1q_u8(block.qs.as_ptr());

        // Unpack 4-bit to 8-bit (low and high nibbles)
        let lo = vreinterpretq_s8_u8(vandq_u8(packed, m4b));
        let hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));

        // Subtract bias (8)
        let lo_s = vsubq_s8(lo, s8b);
        let hi_s = vsubq_s8(hi, s8b);

        // Convert to int32 (sign extend) then to float32
        // Process in 4-element chunks for better NEON utilization
        let lo_0 = vmovl_s8(vget_low_s8(lo_s)); // 8 x int16
        let lo_1 = vmovl_s8(vget_high_s8(lo_s));
        let hi_0 = vmovl_s8(vget_low_s8(hi_s));
        let hi_1 = vmovl_s8(vget_high_s8(hi_s));

        // Convert int16 to float32 and multiply by scale
        // Process lower half of block (first 16 values)
        let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_0)));
        let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_0)));
        let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_1)));
        let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_1)));

        // Apply scale
        let f0 = vmulq_f32(f0, scale_vec);
        let f1 = vmulq_f32(f1, scale_vec);
        let f2 = vmulq_f32(f2, scale_vec);
        let f3 = vmulq_f32(f3, scale_vec);

        // Load y, apply weight, accumulate
        let y0 = vld1q_f32(y_ptr);
        let y1 = vld1q_f32(y_ptr.add(4));
        let y2 = vld1q_f32(y_ptr.add(8));
        let y3 = vld1q_f32(y_ptr.add(12));
        vst1q_f32(y_ptr, vfmaq_f32(y0, f0, w_vec));
        vst1q_f32(y_ptr.add(4), vfmaq_f32(y1, f1, w_vec));
        vst1q_f32(y_ptr.add(8), vfmaq_f32(y2, f2, w_vec));
        vst1q_f32(y_ptr.add(12), vfmaq_f32(y3, f3, w_vec));

        // Process upper half (high nibbles, second 16 values)
        let f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_0)));
        let f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_0)));
        let f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_1)));
        let f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_1)));

        let f4 = vmulq_f32(f4, scale_vec);
        let f5 = vmulq_f32(f5, scale_vec);
        let f6 = vmulq_f32(f6, scale_vec);
        let f7 = vmulq_f32(f7, scale_vec);

        let y4 = vld1q_f32(y_ptr.add(16));
        let y5 = vld1q_f32(y_ptr.add(20));
        let y6 = vld1q_f32(y_ptr.add(24));
        let y7 = vld1q_f32(y_ptr.add(28));
        vst1q_f32(y_ptr.add(16), vfmaq_f32(y4, f4, w_vec));
        vst1q_f32(y_ptr.add(20), vfmaq_f32(y5, f5, w_vec));
        vst1q_f32(y_ptr.add(24), vfmaq_f32(y6, f6, w_vec));
        vst1q_f32(y_ptr.add(28), vfmaq_f32(y7, f7, w_vec));
    }
}

/// Fused vec_mad: y += weight * V (Q4_1 format)
/// Optimized: NEON vectorized dequantization with min offset
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f32_q4_1(
    y: &mut [f32],
    v: &[BlockQ4_1],
    weight: f32,
    n: usize,
) {
    let nb = n / QK;
    let w_vec = vdupq_n_f32(weight);
    let m4b = vdupq_n_u8(0x0F);

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let min = fp16_to_fp32_scalar(block.m);
        let scale_vec = vdupq_n_f32(scale);
        let min_vec = vdupq_n_f32(min);
        let y_ptr = y.as_mut_ptr().add(b * QK);

        // Load 16 bytes of packed 4-bit values
        let packed = vld1q_u8(block.qs.as_ptr());

        // Unpack 4-bit to 8-bit (low and high nibbles)
        let lo = vandq_u8(packed, m4b);
        let hi = vshrq_n_u8(packed, 4);

        // Convert to int16 (zero extend u8 to u16, then reinterpret as i16)
        // Q4_1 values are 0-15, need to convert to float
        let lo_0 = vmovl_u8(vget_low_u8(lo));
        let lo_1 = vmovl_u8(vget_high_u8(lo));
        let hi_0 = vmovl_u8(vget_low_u8(hi));
        let hi_1 = vmovl_u8(vget_high_u8(hi));

        // Convert uint16 to float32
        let f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_0)));
        let f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_0)));
        let f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_1)));
        let f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_1)));

        // Apply scale * val + min
        let f0 = vmlaq_f32(min_vec, f0, scale_vec);
        let f1 = vmlaq_f32(min_vec, f1, scale_vec);
        let f2 = vmlaq_f32(min_vec, f2, scale_vec);
        let f3 = vmlaq_f32(min_vec, f3, scale_vec);

        // Load y, apply weight, accumulate
        let y0 = vld1q_f32(y_ptr);
        let y1 = vld1q_f32(y_ptr.add(4));
        let y2 = vld1q_f32(y_ptr.add(8));
        let y3 = vld1q_f32(y_ptr.add(12));
        vst1q_f32(y_ptr, vfmaq_f32(y0, f0, w_vec));
        vst1q_f32(y_ptr.add(4), vfmaq_f32(y1, f1, w_vec));
        vst1q_f32(y_ptr.add(8), vfmaq_f32(y2, f2, w_vec));
        vst1q_f32(y_ptr.add(12), vfmaq_f32(y3, f3, w_vec));

        // Process upper half (high nibbles)
        let f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_0)));
        let f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_0)));
        let f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_1)));
        let f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_1)));

        let f4 = vmlaq_f32(min_vec, f4, scale_vec);
        let f5 = vmlaq_f32(min_vec, f5, scale_vec);
        let f6 = vmlaq_f32(min_vec, f6, scale_vec);
        let f7 = vmlaq_f32(min_vec, f7, scale_vec);

        let y4 = vld1q_f32(y_ptr.add(16));
        let y5 = vld1q_f32(y_ptr.add(20));
        let y6 = vld1q_f32(y_ptr.add(24));
        let y7 = vld1q_f32(y_ptr.add(28));
        vst1q_f32(y_ptr.add(16), vfmaq_f32(y4, f4, w_vec));
        vst1q_f32(y_ptr.add(20), vfmaq_f32(y5, f5, w_vec));
        vst1q_f32(y_ptr.add(24), vfmaq_f32(y6, f6, w_vec));
        vst1q_f32(y_ptr.add(28), vfmaq_f32(y7, f7, w_vec));
    }
}

/// Lookup table for bit expansion
/// Maps each byte to 8 bytes where bit b becomes (b << 4) in byte position b
static TABLE_B2B_1_Q5_0: [u64; 256] = {
    let mut table = [0u64; 256];
    let mut i = 0;
    while i < 256 {
        let mut val = 0u64;
        let mut j = 0;
        while j < 8 {
            val |= (((i >> j) & 1) as u64) << (j * 8 + 4);
            j += 1;
        }
        table[i] = val;
        i += 1;
    }
    table
};

/// Fused vec_mad: y += weight * V (Q5_0 format)
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f32_q5_0(
    y: &mut [f32],
    v: &[BlockQ5_0],
    weight: f32,
    n: usize,
) {
    let nb = n / QK;
    let m4b = vdupq_n_u8(0x0F);
    let v16 = vdupq_n_s8(16);
    let table_ptr = TABLE_B2B_1_Q5_0.as_ptr();

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let base = b * QK;

        // Load 4-bit quants
        let qs = vld1q_u8(block.qs.as_ptr());

        // Unpack 4-bit to 8-bit
        let qsl = vreinterpretq_s8_u8(vandq_u8(qs, m4b));
        let qsh = vreinterpretq_s8_u8(vshrq_n_u8(qs, 4));

        let mut qh: u32 = 0;
        std::ptr::copy_nonoverlapping(
            block.qh.as_ptr(),
            &mut qh as *mut u32 as *mut u8,
            4,
        );

        let tmp: [u64; 4] = [
            *table_ptr.add(((qh >> 0) & 0xFF) as usize),
            *table_ptr.add(((qh >> 8) & 0xFF) as usize),
            *table_ptr.add(((qh >> 16) & 0xFF) as usize),
            *table_ptr.add((qh >> 24) as usize),
        ];

        // Load expanded bits as int8 vectors
        let qhl = vld1q_s8(tmp.as_ptr() as *const i8);
        let qhh = vld1q_s8(tmp.as_ptr().add(2) as *const i8);

        // Add high bit and subtract 16: (nibble + bit*16) - 16
        let vl_i8 = vsubq_s8(vaddq_s8(qsl, qhl), v16);
        let vh_i8 = vsubq_s8(vaddq_s8(qsh, qhh), v16);

        // Convert to FP32 and accumulate
        // Process low 16 bytes -> 4 vectors of 4 floats
        let vl_0 = vget_low_s8(vl_i8);
        let vl_1 = vget_high_s8(vl_i8);
        let vh_0 = vget_low_s8(vh_i8);
        let vh_1 = vget_high_s8(vh_i8);

        // Convert i8 -> i16 -> i32 -> f32
        let vl_0_i16 = vmovl_s8(vl_0);
        let vl_1_i16 = vmovl_s8(vl_1);
        let vh_0_i16 = vmovl_s8(vh_0);
        let vh_1_i16 = vmovl_s8(vh_1);

        let vl_0_i32 = vmovl_s16(vget_low_s16(vl_0_i16));
        let vl_1_i32 = vmovl_s16(vget_high_s16(vl_0_i16));
        let vl_2_i32 = vmovl_s16(vget_low_s16(vl_1_i16));
        let vl_3_i32 = vmovl_s16(vget_high_s16(vl_1_i16));

        let vh_0_i32 = vmovl_s16(vget_low_s16(vh_0_i16));
        let vh_1_i32 = vmovl_s16(vget_high_s16(vh_0_i16));
        let vh_2_i32 = vmovl_s16(vget_low_s16(vh_1_i16));
        let vh_3_i32 = vmovl_s16(vget_high_s16(vh_1_i16));

        // Convert to f32, scale, multiply by weight, and accumulate
        // y += weight * (scale * value) = (weight * scale) * value
        let ws = weight * scale;
        let ws_vec = vdupq_n_f32(ws);

        // Load y values once
        let y_ptr = y.as_mut_ptr().add(base);
        let y0 = vld1q_f32(y_ptr);
        let y1 = vld1q_f32(y_ptr.add(4));
        let y2 = vld1q_f32(y_ptr.add(8));
        let y3 = vld1q_f32(y_ptr.add(12));
        let y4 = vld1q_f32(y_ptr.add(16));
        let y5 = vld1q_f32(y_ptr.add(20));
        let y6 = vld1q_f32(y_ptr.add(24));
        let y7 = vld1q_f32(y_ptr.add(28));

        // FMLA: y = y + ws * value
        // vfmaq_f32(a, b, c) = a + b * c
        let r0 = vfmaq_f32(y0, vcvtq_f32_s32(vl_0_i32), ws_vec);
        let r1 = vfmaq_f32(y1, vcvtq_f32_s32(vl_1_i32), ws_vec);
        let r2 = vfmaq_f32(y2, vcvtq_f32_s32(vl_2_i32), ws_vec);
        let r3 = vfmaq_f32(y3, vcvtq_f32_s32(vl_3_i32), ws_vec);
        let r4 = vfmaq_f32(y4, vcvtq_f32_s32(vh_0_i32), ws_vec);
        let r5 = vfmaq_f32(y5, vcvtq_f32_s32(vh_1_i32), ws_vec);
        let r6 = vfmaq_f32(y6, vcvtq_f32_s32(vh_2_i32), ws_vec);
        let r7 = vfmaq_f32(y7, vcvtq_f32_s32(vh_3_i32), ws_vec);

        // Store results
        vst1q_f32(y_ptr, r0);
        vst1q_f32(y_ptr.add(4), r1);
        vst1q_f32(y_ptr.add(8), r2);
        vst1q_f32(y_ptr.add(12), r3);
        vst1q_f32(y_ptr.add(16), r4);
        vst1q_f32(y_ptr.add(20), r5);
        vst1q_f32(y_ptr.add(24), r6);
        vst1q_f32(y_ptr.add(28), r7);
    }
}

/// Fused vec_mad: y += weight * V (Q5_1 format)
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f32_q5_1(
    y: &mut [f32],
    v: &[BlockQ5_1],
    weight: f32,
    n: usize,
) {
    let nb = n / QK;
    let m4b = vdupq_n_u8(0x0F);

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let min = fp16_to_fp32_scalar(block.m);
        let base = b * QK;

        // Load 4-bit quants
        let qs = vld1q_u8(block.qs.as_ptr());

        // Unpack 4-bit to 8-bit
        let qsl = vandq_u8(qs, m4b);
        let qsh = vshrq_n_u8(qs, 4);

        // Extract 5th bit (qh is 4 bytes for 32 values)
        let mut qh: u32 = 0;
        std::ptr::copy_nonoverlapping(
            block.qh.as_ptr(),
            &mut qh as *mut u32 as *mut u8,
            4,
        );

        // Expand bits to match value positions
        let tmp: [u64; 4] = [
            *TABLE_B2B_1_Q5_0.as_ptr().add(((qh >> 0) & 0xFF) as usize),
            *TABLE_B2B_1_Q5_0.as_ptr().add(((qh >> 8) & 0xFF) as usize),
            *TABLE_B2B_1_Q5_0.as_ptr().add(((qh >> 16) & 0xFF) as usize),
            *TABLE_B2B_1_Q5_0.as_ptr().add((qh >> 24) as usize),
        ];

        let qhl = vld1q_s8(tmp.as_ptr() as *const i8);
        let qhh = vld1q_s8(tmp.as_ptr().add(2) as *const i8);

        // Combine nibble + high bit: value = nibble + bit*16
        let vl_u8 = vaddq_u8(qsl, vreinterpretq_u8_s8(qhl));
        let vh_u8 = vaddq_u8(qsh, vreinterpretq_u8_s8(qhh));

        // Convert to f32
        let vl_0 = vmovl_u8(vget_low_u8(vl_u8));
        let vl_1 = vmovl_u8(vget_high_u8(vl_u8));
        let vh_0 = vmovl_u8(vget_low_u8(vh_u8));
        let vh_1 = vmovl_u8(vget_high_u8(vh_u8));

        let f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vl_0)));
        let f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vl_0)));
        let f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vl_1)));
        let f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vl_1)));
        let f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vh_0)));
        let f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vh_0)));
        let f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vh_1)));
        let f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vh_1)));

        // Apply scale and min: value = scale * (nibble + bit*16) + min
        let scale_vec = vdupq_n_f32(scale);
        let min_vec = vdupq_n_f32(min);
        let f0 = vmlaq_f32(min_vec, f0, scale_vec);
        let f1 = vmlaq_f32(min_vec, f1, scale_vec);
        let f2 = vmlaq_f32(min_vec, f2, scale_vec);
        let f3 = vmlaq_f32(min_vec, f3, scale_vec);
        let f4 = vmlaq_f32(min_vec, f4, scale_vec);
        let f5 = vmlaq_f32(min_vec, f5, scale_vec);
        let f6 = vmlaq_f32(min_vec, f6, scale_vec);
        let f7 = vmlaq_f32(min_vec, f7, scale_vec);

        // Apply weight and accumulate
        let w_vec = vdupq_n_f32(weight);
        let y_ptr = y.as_mut_ptr().add(base);
        let y0 = vld1q_f32(y_ptr);
        let y1 = vld1q_f32(y_ptr.add(4));
        let y2 = vld1q_f32(y_ptr.add(8));
        let y3 = vld1q_f32(y_ptr.add(12));
        let y4 = vld1q_f32(y_ptr.add(16));
        let y5 = vld1q_f32(y_ptr.add(20));
        let y6 = vld1q_f32(y_ptr.add(24));
        let y7 = vld1q_f32(y_ptr.add(28));

        vst1q_f32(y_ptr, vfmaq_f32(y0, f0, w_vec));
        vst1q_f32(y_ptr.add(4), vfmaq_f32(y1, f1, w_vec));
        vst1q_f32(y_ptr.add(8), vfmaq_f32(y2, f2, w_vec));
        vst1q_f32(y_ptr.add(12), vfmaq_f32(y3, f3, w_vec));
        vst1q_f32(y_ptr.add(16), vfmaq_f32(y4, f4, w_vec));
        vst1q_f32(y_ptr.add(20), vfmaq_f32(y5, f5, w_vec));
        vst1q_f32(y_ptr.add(24), vfmaq_f32(y6, f6, w_vec));
        vst1q_f32(y_ptr.add(28), vfmaq_f32(y7, f7, w_vec));
    }
}

/// Fused vec_mad: y += weight * V (IQ4_NL format)
/// Optimized: use KVALUES lookup table for dequantization
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f32_iq4_nl(
    y: &mut [f32],
    v: &[BlockIQ4NL],
    weight: f32,
    n: usize,
) {
    use crate::quant::kv_cache::KVALUES_IQ4NL_I8;

    let nb = n / QK;
    let m4b = vdupq_n_u8(0x0F);
    let kvalues = KVALUES_IQ4NL_I8.as_ptr();

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let base = b * QK;

        // Load 16 bytes of packed 4-bit values
        let packed = vld1q_u8(block.qs.as_ptr());

        // Unpack 4-bit to 8-bit (low and high nibbles)
        let lo = vandq_u8(packed, m4b);
        let hi = vshrq_n_u8(packed, 4);

        // Lookup kvalues for each nibble using TBL
        // Build lookup table in vector registers
        // Since TBL uses 16-byte table, we load kvalues[0..16]
        let lut = vld1q_s8(kvalues);

        // TBL: table lookup for each byte
        let lo_i8 = vqtbl1q_s8(lut, lo);
        let hi_i8 = vqtbl1q_s8(lut, hi);

        // Convert to f32
        let lo_0 = vmovl_s8(vget_low_s8(lo_i8));
        let lo_1 = vmovl_s8(vget_high_s8(lo_i8));
        let hi_0 = vmovl_s8(vget_low_s8(hi_i8));
        let hi_1 = vmovl_s8(vget_high_s8(hi_i8));

        let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_0)));
        let f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_0)));
        let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_1)));
        let f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_1)));
        let f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_0)));
        let f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_0)));
        let f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_1)));
        let f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_1)));

        // Apply scale
        let scale_vec = vdupq_n_f32(scale);
        let f0 = vmulq_f32(f0, scale_vec);
        let f1 = vmulq_f32(f1, scale_vec);
        let f2 = vmulq_f32(f2, scale_vec);
        let f3 = vmulq_f32(f3, scale_vec);
        let f4 = vmulq_f32(f4, scale_vec);
        let f5 = vmulq_f32(f5, scale_vec);
        let f6 = vmulq_f32(f6, scale_vec);
        let f7 = vmulq_f32(f7, scale_vec);

        // Apply weight and accumulate
        let w_vec = vdupq_n_f32(weight);
        let y_ptr = y.as_mut_ptr().add(base);
        let y0 = vld1q_f32(y_ptr);
        let y1 = vld1q_f32(y_ptr.add(4));
        let y2 = vld1q_f32(y_ptr.add(8));
        let y3 = vld1q_f32(y_ptr.add(12));
        let y4 = vld1q_f32(y_ptr.add(16));
        let y5 = vld1q_f32(y_ptr.add(20));
        let y6 = vld1q_f32(y_ptr.add(24));
        let y7 = vld1q_f32(y_ptr.add(28));

        vst1q_f32(y_ptr, vfmaq_f32(y0, f0, w_vec));
        vst1q_f32(y_ptr.add(4), vfmaq_f32(y1, f1, w_vec));
        vst1q_f32(y_ptr.add(8), vfmaq_f32(y2, f2, w_vec));
        vst1q_f32(y_ptr.add(12), vfmaq_f32(y3, f3, w_vec));
        vst1q_f32(y_ptr.add(16), vfmaq_f32(y4, f4, w_vec));
        vst1q_f32(y_ptr.add(20), vfmaq_f32(y5, f5, w_vec));
        vst1q_f32(y_ptr.add(24), vfmaq_f32(y6, f6, w_vec));
        vst1q_f32(y_ptr.add(28), vfmaq_f32(y7, f7, w_vec));
    }
}
