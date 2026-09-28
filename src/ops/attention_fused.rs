// Fused vec_dot and vec_mad for KV cache quantization formats
// Following llama.cpp's approach: quantize Q to Q8_0, then use Q8_0 × Qn_0 NEON fused vec_dot
// This enables hardware INT8 dot product acceleration (SDOT instruction)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::quant::kv_cache::quant::{
    quantize_row_q8_0_neon, quantize_row_q8_1,
};
use crate::quant::kv_cache::{
    BlockIQ4NL, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0,
    BlockQ8_1, KVALUES_IQ4NL, KVALUES_IQ4NL_I8, QK,
};

// ============================================================================
// Q8_0 Quantization for Query
// ============================================================================

/// Quantize FP32 query to Q8_0 for use in fused vec_dot
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn quantize_query_to_q8_0(q: &[f32], q_q8: &mut [BlockQ8_0]) {
    quantize_row_q8_0_neon(q, q_q8);
}

/// Quantize FP32 query to Q8_1 for use in Q4_1/Q5_1 fused vec_dot
#[cfg(target_arch = "aarch64")]
pub fn quantize_query_to_q8_1(q: &[f32], q_q8: &mut [BlockQ8_1]) {
    quantize_row_q8_1(q, q_q8);
}

// ============================================================================
// Helper Functions
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn fp16_to_fp32_scalar(bits: u16) -> f32 {
    let f16 = vreinterpret_f16_u16(vdup_n_u16(bits));
    let f32_vec = vcvt_f32_f16(f16);
    vgetq_lane_f32(f32_vec, 0)
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn bf16_to_fp32_scalar(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ============================================================================
// Vec Dot: Q8_0 × Qn_0 (direct transmute, no allocation)
// ============================================================================

/// Q8_0 × Q8_0 fused vec_dot (using types from quant/types)
/// Direct transmute - no allocation
#[cfg(target_arch = "aarch64")]
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
#[cfg(target_arch = "aarch64")]
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
#[cfg(target_arch = "aarch64")]
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
#[cfg(target_arch = "aarch64")]
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
#[cfg(target_arch = "aarch64")]
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
#[cfg(target_arch = "aarch64")]
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
#[cfg(target_arch = "aarch64")]
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
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f32_q8_0_neon(
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
/// Optimized: scalar dequant + NEON FMA (simple and fast)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f32_q4_0_neon(
    y: &mut [f32],
    v: &[BlockQ4_0],
    weight: f32,
    n: usize,
) {
    let nb = n / QK;

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);

        // Scalar dequantization (memory-bound, SIMD doesn't help much)
        let mut vals = [0.0f32; QK];
        for j in 0..(QK / 2) {
            let packed = block.qs[j];
            let lo = (packed & 0x0F) as i32 - 8;
            let hi = ((packed >> 4) & 0x0F) as i32 - 8;
            vals[j] = (lo as f32) * scale;
            vals[j + QK / 2] = (hi as f32) * scale;
        }

        // NEON FMA: y += weight * vals
        let w_vec = vdupq_n_f32(weight);
        let y_ptr = y.as_mut_ptr().add(b * QK);
        for i in (0..QK).step_by(4) {
            let y_vec = vld1q_f32(y_ptr.add(i));
            let v_vec = vld1q_f32(vals.as_ptr().add(i));
            vst1q_f32(y_ptr.add(i), vfmaq_f32(y_vec, v_vec, w_vec));
        }
    }
}

/// Fused vec_mad: y += weight * V (Q4_1 format)
/// Optimized: scalar dequant + NEON accumulation (matches llama.cpp approach)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f32_q4_1_neon(
    y: &mut [f32],
    v: &[BlockQ4_1],
    weight: f32,
    n: usize,
) {
    let nb = n / QK;

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let min = fp16_to_fp32_scalar(block.m);

        // Scalar dequantization
        let mut vals = [0.0f32; QK];
        for j in 0..(QK / 2) {
            let packed = block.qs[j];
            let lo = (packed & 0x0F) as f32;
            let hi = ((packed >> 4) & 0x0F) as f32;

            vals[j] = scale * lo + min;
            vals[j + QK / 2] = scale * hi + min;
        }

        // NEON accumulation: y += weight * vals
        let weight_vec = vdupq_n_f32(weight);
        let base = b * QK;

        for i in (0..QK).step_by(4) {
            let y_vec = vld1q_f32(y.as_ptr().add(base + i));
            let v_vec = vld1q_f32(vals.as_ptr().add(i));
            let result = vmlaq_f32(y_vec, v_vec, weight_vec);
            vst1q_f32(y.as_mut_ptr().add(base + i), result);
        }
    }
}

/// Lookup table for bit expansion
/// Maps each byte to 8 bytes where bit b becomes (b << 4) in byte position b
#[cfg(target_arch = "aarch64")]
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
/// Optimized: use lookup table for qh bit expansion (matches llama.cpp exactly)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f32_q5_0_neon(
    y: &mut [f32],
    v: &[BlockQ5_0],
    weight: f32,
    n: usize,
) {
    let nb = n / QK;
    let m4b = vdupq_n_u8(0x0F);
    let v16 = vdupq_n_s8(16);
    let weight_vec = vdupq_n_f32(weight);
    let table_ptr = TABLE_B2B_1_Q5_0.as_ptr();

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let scale_vec = vdupq_n_f32(scale);
        let base = b * QK;

        // Load 4-bit quants
        let qs = vld1q_u8(block.qs.as_ptr());

        // Unpack 4-bit to 8-bit
        let qsl = vreinterpretq_s8_u8(vandq_u8(qs, m4b));
        let qsh = vreinterpretq_s8_u8(vshrq_n_u8(qs, 4));

        // Extract 5th bit via lookup table (use memcpy like llama.cpp)
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
        // y += weight * (scale * value)
        let vl_0_f32 = vmulq_f32(vcvtq_f32_s32(vl_0_i32), scale_vec);
        let vl_1_f32 = vmulq_f32(vcvtq_f32_s32(vl_1_i32), scale_vec);
        let vl_2_f32 = vmulq_f32(vcvtq_f32_s32(vl_2_i32), scale_vec);
        let vl_3_f32 = vmulq_f32(vcvtq_f32_s32(vl_3_i32), scale_vec);

        let vh_0_f32 = vmulq_f32(vcvtq_f32_s32(vh_0_i32), scale_vec);
        let vh_1_f32 = vmulq_f32(vcvtq_f32_s32(vh_1_i32), scale_vec);
        let vh_2_f32 = vmulq_f32(vcvtq_f32_s32(vh_2_i32), scale_vec);
        let vh_3_f32 = vmulq_f32(vcvtq_f32_s32(vh_3_i32), scale_vec);

        // Load y and accumulate: y += weight * v
        vst1q_f32(
            y.as_mut_ptr().add(base + 0),
            vmlaq_f32(
                vld1q_f32(y.as_ptr().add(base + 0)),
                vl_0_f32,
                weight_vec,
            ),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 4),
            vmlaq_f32(
                vld1q_f32(y.as_ptr().add(base + 4)),
                vl_1_f32,
                weight_vec,
            ),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 8),
            vmlaq_f32(
                vld1q_f32(y.as_ptr().add(base + 8)),
                vl_2_f32,
                weight_vec,
            ),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 12),
            vmlaq_f32(
                vld1q_f32(y.as_ptr().add(base + 12)),
                vl_3_f32,
                weight_vec,
            ),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 16),
            vmlaq_f32(
                vld1q_f32(y.as_ptr().add(base + 16)),
                vh_0_f32,
                weight_vec,
            ),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 20),
            vmlaq_f32(
                vld1q_f32(y.as_ptr().add(base + 20)),
                vh_1_f32,
                weight_vec,
            ),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 24),
            vmlaq_f32(
                vld1q_f32(y.as_ptr().add(base + 24)),
                vh_2_f32,
                weight_vec,
            ),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 28),
            vmlaq_f32(
                vld1q_f32(y.as_ptr().add(base + 28)),
                vh_3_f32,
                weight_vec,
            ),
        );
    }
}

/// Fused vec_mad: y += weight * V (Q5_1 format)
/// Optimized: NEON for parallel unpack + accumulation (matches llama.cpp approach)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f32_q5_1_neon(
    y: &mut [f32],
    v: &[BlockQ5_1],
    weight: f32,
    n: usize,
) {
    let nb = n / QK;

    let table_ptr = TABLE_B2B_1_Q5_0.as_ptr();

    let m4b = vdupq_n_u8(0x0F);
    let v16 = vdupq_n_s8(16);
    let weight_vec = vdupq_n_f32(weight);

    for b in 0..nb {
        let block = &v[b];
        let scale = fp16_to_fp32_scalar(block.d);
        let min = fp16_to_fp32_scalar(block.m);
        let scale_vec = vdupq_n_f32(scale);
        let min_vec = vdupq_n_f32(min);
        let base = b * QK;

        // Load 4-bit quants
        let qs = vld1q_u8(block.qs.as_ptr());

        // Unpack 4-bit to 8-bit
        let qsl = vreinterpretq_s8_u8(vandq_u8(qs, m4b));
        let qsh = vreinterpretq_s8_u8(vshrq_n_u8(qs, 4));

        // Extract 5th bit via lookup table
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

        let qhl = vld1q_s8(tmp.as_ptr() as *const i8);
        let qhh = vld1q_s8(tmp.as_ptr().add(2) as *const i8);

        // Q5_1: value = scale * q + min, where q is 0-31
        // q = nibble + bit*16, no bias subtraction needed (unlike Q5_0)
        let vl_i8 = vaddq_s8(qsl, qhl); // nibble + bit*16
        let vh_i8 = vaddq_s8(qsh, qhh);

        // Convert i8 -> i16 -> i32 -> f32
        let vl_0_i16 = vmovl_s8(vget_low_s8(vl_i8));
        let vl_1_i16 = vmovl_s8(vget_high_s8(vl_i8));
        let vh_0_i16 = vmovl_s8(vget_low_s8(vh_i8));
        let vh_1_i16 = vmovl_s8(vget_high_s8(vh_i8));

        let vl_0_i32 = vmovl_s16(vget_low_s16(vl_0_i16));
        let vl_1_i32 = vmovl_s16(vget_high_s16(vl_0_i16));
        let vl_2_i32 = vmovl_s16(vget_low_s16(vl_1_i16));
        let vl_3_i32 = vmovl_s16(vget_high_s16(vl_1_i16));

        let vh_0_i32 = vmovl_s16(vget_low_s16(vh_0_i16));
        let vh_1_i32 = vmovl_s16(vget_high_s16(vh_0_i16));
        let vh_2_i32 = vmovl_s16(vget_low_s16(vh_1_i16));
        let vh_3_i32 = vmovl_s16(vget_high_s16(vh_1_i16));

        // Convert to f32: value = scale * q + min
        // y += weight * (scale * q + min) = weight * scale * q + weight * min
        // Optimize: pre-compute weight * scale and weight * min
        let ws_vec = vmulq_n_f32(scale_vec, weight);
        let wm_vec = vmulq_n_f32(min_vec, weight);

        let vl_0 = vmlaq_f32(wm_vec, vcvtq_f32_s32(vl_0_i32), ws_vec);
        let vl_1 = vmlaq_f32(wm_vec, vcvtq_f32_s32(vl_1_i32), ws_vec);
        let vl_2 = vmlaq_f32(wm_vec, vcvtq_f32_s32(vl_2_i32), ws_vec);
        let vl_3 = vmlaq_f32(wm_vec, vcvtq_f32_s32(vl_3_i32), ws_vec);

        let vh_0 = vmlaq_f32(wm_vec, vcvtq_f32_s32(vh_0_i32), ws_vec);
        let vh_1 = vmlaq_f32(wm_vec, vcvtq_f32_s32(vh_1_i32), ws_vec);
        let vh_2 = vmlaq_f32(wm_vec, vcvtq_f32_s32(vh_2_i32), ws_vec);
        let vh_3 = vmlaq_f32(wm_vec, vcvtq_f32_s32(vh_3_i32), ws_vec);

        // Accumulate: y += value
        vst1q_f32(
            y.as_mut_ptr().add(base + 0),
            vaddq_f32(vld1q_f32(y.as_ptr().add(base + 0)), vl_0),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 4),
            vaddq_f32(vld1q_f32(y.as_ptr().add(base + 4)), vl_1),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 8),
            vaddq_f32(vld1q_f32(y.as_ptr().add(base + 8)), vl_2),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 12),
            vaddq_f32(vld1q_f32(y.as_ptr().add(base + 12)), vl_3),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 16),
            vaddq_f32(vld1q_f32(y.as_ptr().add(base + 16)), vh_0),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 20),
            vaddq_f32(vld1q_f32(y.as_ptr().add(base + 20)), vh_1),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 24),
            vaddq_f32(vld1q_f32(y.as_ptr().add(base + 24)), vh_2),
        );
        vst1q_f32(
            y.as_mut_ptr().add(base + 28),
            vaddq_f32(vld1q_f32(y.as_ptr().add(base + 28)), vh_3),
        );
    }
}

/// Fused vec_mad: y += weight * V (IQ4_NL format)
/// Strategy: scalar dequant + NEON FMA (matches llama.cpp approach)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_mad_f32_iq4_nl_neon(
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
        let base = b * QK;

        // Scalar dequantization (matches llama.cpp's approach)
        // This is memory-bound, SIMD widening overhead doesn't help
        let mut vals = [0.0f32; QK];
        for j in 0..(QK / 2) {
            let packed = block.qs[j];
            let lo = (packed & 0x0F) as usize;
            let hi = ((packed >> 4) & 0x0F) as usize;

            vals[j] = KVALUES_IQ4NL[lo];
            vals[j + QK / 2] = KVALUES_IQ4NL[hi];
        }

        // NEON FMA: y += ws * vals
        let ws_vec = vdupq_n_f32(ws);
        for i in (0..QK).step_by(4) {
            let y_vec = vld1q_f32(y.as_ptr().add(base + i));
            let v_vec = vld1q_f32(vals.as_ptr().add(i));
            vst1q_f32(
                y.as_mut_ptr().add(base + i),
                vfmaq_f32(y_vec, v_vec, ws_vec),
            );
        }
    }
}
