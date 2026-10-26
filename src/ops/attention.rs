// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Attention operator implementation with fused KV dequantization

#![allow(asm_sub_register, dead_code)]

#[cfg(target_arch = "aarch64")]
use crate::ops::arm::neon::{
    vaddq_f16_asm, vcvt_f16_from_f32, vcvt_f32_from_f16, vfmaq_f16_asm,
    vld1_f16_and_cvt_to_f32, vmulq_f16_asm,
};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::asm;

use crate::ops::softmax::softmax_f32_dispatch;
use crate::quant::kv_cache::{
    BlockIQ4NL, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0,
    BlockQ8_1, QK,
};

// ============================================================================
// FP32 and FP16 Attention (baseline)
// ============================================================================

/// Decode attention for batch=1, FP32 version (scalar baseline)
pub fn decode_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    for h in 0..num_heads {
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

        let mut scores = vec![0.0f32; seq_len];

        for s in 0..seq_len {
            let k_row = &k[(s * num_heads + h) * head_dim
                ..(s * num_heads + h + 1) * head_dim];
            scores[s] = vec_dot_f32_scalar(q_head, k_row) * scale;
        }

        let mut attn_weights = vec![0.0f32; seq_len];
        softmax_f32_dispatch(&scores, &mut attn_weights);

        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for s in 0..seq_len {
                sum += attn_weights[s] * v[(s * num_heads + h) * head_dim + d];
            }
            output_head[d] = sum;
        }
    }
}

// ============================================================================
// BF16 Attention
// ============================================================================

/// Decode attention for batch=1, BF16 version with NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn decode_attention_bf16_neon(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    let mut k_f32 = vec![0.0f32; head_dim];

    for h in 0..num_heads {
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

        let mut max_val = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;

        for d in 0..head_dim {
            output_head[d] = 0.0;
        }

        for s in 0..seq_len {
            let k_offset = (s * num_heads + h) * head_dim;
            dequantize_row_bf16_neon_inline(
                &k[k_offset..k_offset + head_dim],
                &mut k_f32,
            );

            let score = vec_dot_f32_neon(q_head, &k_f32, head_dim) * scale;

            let old_max = max_val;
            if score > max_val {
                max_val = score;
            }

            let weight;
            if score > old_max {
                let correction = (old_max - max_val).exp();
                sum_exp *= correction;
                // NEON vector scaling
                let corr_vec = vdupq_n_f32(correction);
                for d in (0..head_dim).step_by(4) {
                    if d + 4 <= head_dim {
                        let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                        let scaled = vmulq_f32(out_vec, corr_vec);
                        vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
                    } else {
                        for dd in d..head_dim {
                            output_head[dd] *= correction;
                        }
                    }
                }
                weight = 1.0;
            } else {
                weight = (score - max_val).exp();
            }

            sum_exp += weight;

            let v_offset = (s * num_heads + h) * head_dim;
            for d in (0..head_dim).step_by(4) {
                if d + 4 <= head_dim {
                    let v_bf16 = vld1_u16(v.as_ptr().add(v_offset + d));
                    let v_u32 = vmovl_u16(v_bf16);
                    let v_f32 = vreinterpretq_f32_u32(vshlq_n_u32(v_u32, 16));
                    let w_vec = vdupq_n_f32(weight);
                    let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                    let result = vfmaq_f32(out_vec, v_f32, w_vec);
                    vst1q_f32(output_head.as_mut_ptr().add(d), result);
                } else {
                    for dd in d..head_dim {
                        let v_f32_val = bf16_to_fp32(v[v_offset + dd]);
                        output_head[dd] += weight * v_f32_val;
                    }
                }
            }
        }

        let inv_sum = 1.0 / sum_exp;
        for d in (0..head_dim).step_by(4) {
            if d + 4 <= head_dim {
                let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                let scaled = vmulq_f32(out_vec, vdupq_n_f32(inv_sum));
                vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
            } else {
                for dd in d..head_dim {
                    output_head[dd] *= inv_sum;
                }
            }
        }
    }
}

// ============================================================================
// Q8_0 Attention
// ============================================================================

/// Decode attention for batch=1, Q8_0 K/V with NEON
/// Strategy: Quantize Q to Q8_0, then use Q8_0 × Q8_0 fused vec_dot
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn decode_attention_q8_0_neon(
    q: &[f32],
    k: &[BlockQ8_0],
    v: &[BlockQ8_0],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    let blocks_per_row = head_dim / QK;

    // Pre-quantize Q buffer
    let mut q_q8 = vec![BlockQ8_0 { d: 0, qs: [0i8; QK] }; blocks_per_row];

    // Zero all output at once with NEON
    for d in (0..num_heads * head_dim).step_by(4) {
        vst1q_f32(output.as_mut_ptr().add(d), vdupq_n_f32(0.0));
    }

    for h in 0..num_heads {
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

        // Quantize Q to Q8_0 (once per head)
        quantize_query_to_q8_0(q_head, &mut q_q8);

        let mut max_val = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;

        for s in 0..seq_len {
            // Q8_0 × Q8_0 NEON vec_dot
            let k_offset = (s * num_heads + h) * blocks_per_row;
            let score = {
                // Transmute to types module blocks (same repr(C))
                let q_ptr =
                    q_q8.as_ptr() as *const crate::quant::types::BlockQ8_0;
                let q_slice = std::slice::from_raw_parts(q_ptr, blocks_per_row);
                let k_ptr = k.as_ptr().add(k_offset)
                    as *const crate::quant::types::BlockQ8_0;
                let k_slice = std::slice::from_raw_parts(k_ptr, blocks_per_row);
                crate::quant::vec_dot::arm::q_0_1::vec_dot_q8_0_q8_0_neon(
                    head_dim, q_slice, k_slice,
                )
            } * scale;

            let old_max = max_val;
            if score > max_val {
                max_val = score;
            }

            let weight;
            if score > old_max {
                let correction = (old_max - max_val).exp();
                sum_exp *= correction;
                // NEON vector scaling
                let corr_vec = vdupq_n_f32(correction);
                for d in (0..head_dim).step_by(4) {
                    let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                    vst1q_f32(
                        output_head.as_mut_ptr().add(d),
                        vmulq_f32(out_vec, corr_vec),
                    );
                }
                weight = 1.0;
            } else {
                weight = (score - max_val).exp();
            }

            sum_exp += weight;

            // Accumulate V with NEON vec_mad
            let v_offset = (s * num_heads + h) * blocks_per_row;
            vec_mad_f32_q8_0(
                output_head,
                &v[v_offset..v_offset + blocks_per_row],
                weight,
                head_dim,
            );
        }

        let inv_sum = 1.0 / sum_exp;
        for d in (0..head_dim).step_by(4) {
            let out_vec = vld1q_f32(output_head.as_ptr().add(d));
            vst1q_f32(
                output_head.as_mut_ptr().add(d),
                vmulq_f32(out_vec, vdupq_n_f32(inv_sum)),
            );
        }
    }
}

// ============================================================================
// Q4_0 Attention
// ============================================================================

/// Decode attention for batch=1, Q4_0 K/V with NEON
/// Strategy: Quantize Q to Q8_0, then use Q8_0 × Q4_0 fused vec_dot
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn decode_attention_q4_0_neon(
    q: &[f32],
    k: &[BlockQ4_0],
    v: &[BlockQ4_0],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    let blocks_per_row = head_dim / QK;

    // Pre-quantize Q buffer
    let mut q_q8 = vec![BlockQ8_0 { d: 0, qs: [0i8; QK] }; blocks_per_row];

    for h in 0..num_heads {
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

        // Quantize Q to Q8_0
        quantize_query_to_q8_0(q_head, &mut q_q8);

        let mut max_val = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;

        for d in 0..head_dim {
            output_head[d] = 0.0;
        }

        for s in 0..seq_len {
            // Q8_0 × Q4_0 fused vec_dot
            let k_offset = (s * num_heads + h) * blocks_per_row;
            let score = vec_dot_q4_0_q8_0(
                head_dim,
                &q_q8,
                &k[k_offset..k_offset + blocks_per_row],
            ) * scale;

            let old_max = max_val;
            if score > max_val {
                max_val = score;
            }

            let weight;
            if score > old_max {
                let correction = (old_max - max_val).exp();
                sum_exp *= correction;
                // NEON vector scaling
                let corr_vec = vdupq_n_f32(correction);
                for d in (0..head_dim).step_by(4) {
                    if d + 4 <= head_dim {
                        let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                        let scaled = vmulq_f32(out_vec, corr_vec);
                        vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
                    } else {
                        for dd in d..head_dim {
                            output_head[dd] *= correction;
                        }
                    }
                }
                weight = 1.0;
            } else {
                weight = (score - max_val).exp();
            }

            sum_exp += weight;

            // Accumulate V
            let v_offset = (s * num_heads + h) * blocks_per_row;
            vec_mad_f32_q4_0(
                output_head,
                &v[v_offset..v_offset + blocks_per_row],
                weight,
                head_dim,
            );
        }

        let inv_sum = 1.0 / sum_exp;
        for d in (0..head_dim).step_by(4) {
            if d + 4 <= head_dim {
                let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                let scaled = vmulq_f32(out_vec, vdupq_n_f32(inv_sum));
                vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
            } else {
                for dd in d..head_dim {
                    output_head[dd] *= inv_sum;
                }
            }
        }
    }
}

// ============================================================================
// Q4_1 Attention
// ============================================================================

/// Decode attention for batch=1, Q4_1 K/V with NEON
/// Strategy: Quantize Q to Q8_1, then use Q8_1 × Q4_1 fused vec_dot with NEON SDOT
/// Formula: sum = d_q * d_k * sum(q_i * k_i) + s_q * m_k
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn decode_attention_q4_1_neon(
    q: &[f32],
    k: &[BlockQ4_1],
    v: &[BlockQ4_1],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    let blocks_per_row = head_dim / QK;

    // Quantize Q to Q8_1 (with sum field)
    let mut q_q8 =
        vec![BlockQ8_1 { d: 0, s: 0, qs: [0i8; QK] }; blocks_per_row];

    for h in 0..num_heads {
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

        // Quantize Q to Q8_1
        quantize_query_to_q8_1(q_head, &mut q_q8);

        let mut max_val = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;

        for d in 0..head_dim {
            output_head[d] = 0.0;
        }

        for s in 0..seq_len {
            let k_offset = (s * num_heads + h) * blocks_per_row;

            // Use NEON vec_dot: K is Q4_1, Q is Q8_1
            // The function expects (K: Q4_1, Q: Q8_1) which matches our data
            let score = {
                // Transmute to types module blocks (same repr(C))
                let k_ptr = k.as_ptr().add(k_offset)
                    as *const crate::quant::types::BlockQ4_1;
                let k_slice = std::slice::from_raw_parts(k_ptr, blocks_per_row);
                let q_ptr =
                    q_q8.as_ptr() as *const crate::quant::types::BlockQ8_1;
                let q_slice = std::slice::from_raw_parts(q_ptr, blocks_per_row);
                crate::quant::vec_dot::arm::q_0_1::vec_dot_q4_1_q8_1_neon(
                    head_dim, k_slice, q_slice,
                )
            } * scale;

            let old_max = max_val;
            if score > max_val {
                max_val = score;
            }

            let weight;
            if score > old_max {
                let correction = (old_max - max_val).exp();
                sum_exp *= correction;
                // NEON vector scaling
                let corr_vec = vdupq_n_f32(correction);
                for d in (0..head_dim).step_by(4) {
                    if d + 4 <= head_dim {
                        let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                        let scaled = vmulq_f32(out_vec, corr_vec);
                        vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
                    } else {
                        for dd in d..head_dim {
                            output_head[dd] *= correction;
                        }
                    }
                }
                weight = 1.0;
            } else {
                weight = (score - max_val).exp();
            }

            sum_exp += weight;

            // Accumulate V with NEON vec_mad
            let v_offset = (s * num_heads + h) * blocks_per_row;
            vec_mad_f32_q4_1(
                output_head,
                &v[v_offset..v_offset + blocks_per_row],
                weight,
                head_dim,
            );
        }

        let inv_sum = 1.0 / sum_exp;
        for d in (0..head_dim).step_by(4) {
            if d + 4 <= head_dim {
                let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                let scaled = vmulq_f32(out_vec, vdupq_n_f32(inv_sum));
                vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
            } else {
                for dd in d..head_dim {
                    output_head[dd] *= inv_sum;
                }
            }
        }
    }
}

// ============================================================================
// Q5_0 Attention
// ============================================================================

/// Decode attention for batch=1, Q5_0 K/V with NEON
/// Strategy: Quantize Q to Q8_0, then use Q8_0 × Q5_0 fused vec_dot
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn decode_attention_q5_0_neon(
    q: &[f32],
    k: &[BlockQ5_0],
    v: &[BlockQ5_0],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    let blocks_per_row = head_dim / QK;

    let mut q_q8 = vec![BlockQ8_0 { d: 0, qs: [0i8; QK] }; blocks_per_row];

    for h in 0..num_heads {
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

        quantize_query_to_q8_0(q_head, &mut q_q8);

        let mut max_val = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;

        for d in 0..head_dim {
            output_head[d] = 0.0;
        }

        for s in 0..seq_len {
            let k_offset = (s * num_heads + h) * blocks_per_row;
            let score = vec_dot_q5_0_q8_0(
                head_dim,
                &q_q8,
                &k[k_offset..k_offset + blocks_per_row],
            ) * scale;

            let old_max = max_val;
            if score > max_val {
                max_val = score;
            }

            let weight;
            if score > old_max {
                let correction = (old_max - max_val).exp();
                sum_exp *= correction;
                // NEON vector scaling
                let corr_vec = vdupq_n_f32(correction);
                for d in (0..head_dim).step_by(4) {
                    if d + 4 <= head_dim {
                        let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                        let scaled = vmulq_f32(out_vec, corr_vec);
                        vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
                    } else {
                        for dd in d..head_dim {
                            output_head[dd] *= correction;
                        }
                    }
                }
                weight = 1.0;
            } else {
                weight = (score - max_val).exp();
            }

            sum_exp += weight;

            let v_offset = (s * num_heads + h) * blocks_per_row;
            vec_mad_f32_q5_0(
                output_head,
                &v[v_offset..v_offset + blocks_per_row],
                weight,
                head_dim,
            );
        }

        let inv_sum = 1.0 / sum_exp;
        for d in (0..head_dim).step_by(4) {
            if d + 4 <= head_dim {
                let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                let scaled = vmulq_f32(out_vec, vdupq_n_f32(inv_sum));
                vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
            } else {
                for dd in d..head_dim {
                    output_head[dd] *= inv_sum;
                }
            }
        }
    }
}

// ============================================================================
// Q5_1 Attention
// ============================================================================

/// Decode attention for batch=1, Q5_1 K/V with NEON
/// Strategy: Quantize Q to Q8_1, then use Q8_1 × Q5_1 fused vec_dot with NEON SDOT
/// Formula: sum = d_q * d_k * sum(q_i * k_i) + s_q * m_k
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn decode_attention_q5_1_neon(
    q: &[f32],
    k: &[BlockQ5_1],
    v: &[BlockQ5_1],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    let blocks_per_row = head_dim / QK;

    // Quantize Q to Q8_1 (with sum field)
    let mut q_q8 =
        vec![BlockQ8_1 { d: 0, s: 0, qs: [0i8; QK] }; blocks_per_row];

    for h in 0..num_heads {
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

        // Quantize Q to Q8_1
        quantize_query_to_q8_1(q_head, &mut q_q8);

        let mut max_val = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;

        for d in 0..head_dim {
            output_head[d] = 0.0;
        }

        for s in 0..seq_len {
            let k_offset = (s * num_heads + h) * blocks_per_row;

            // Use NEON vec_dot: K is Q5_1, Q is Q8_1
            let score = {
                // Transmute to types module blocks (same repr(C))
                let k_ptr = k.as_ptr().add(k_offset)
                    as *const crate::quant::types::BlockQ5_1;
                let k_slice = std::slice::from_raw_parts(k_ptr, blocks_per_row);
                let q_ptr =
                    q_q8.as_ptr() as *const crate::quant::types::BlockQ8_1;
                let q_slice = std::slice::from_raw_parts(q_ptr, blocks_per_row);
                crate::quant::vec_dot::arm::q_0_1::vec_dot_q5_1_q8_1_neon(
                    head_dim, k_slice, q_slice,
                )
            } * scale;

            let old_max = max_val;
            if score > max_val {
                max_val = score;
            }

            let weight;
            if score > old_max {
                let correction = (old_max - max_val).exp();
                sum_exp *= correction;
                // NEON vector scaling
                let corr_vec = vdupq_n_f32(correction);
                for d in (0..head_dim).step_by(4) {
                    if d + 4 <= head_dim {
                        let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                        let scaled = vmulq_f32(out_vec, corr_vec);
                        vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
                    } else {
                        for dd in d..head_dim {
                            output_head[dd] *= correction;
                        }
                    }
                }
                weight = 1.0;
            } else {
                weight = (score - max_val).exp();
            }

            sum_exp += weight;

            // Accumulate V with NEON vec_mad
            let v_offset = (s * num_heads + h) * blocks_per_row;
            vec_mad_f32_q5_1(
                output_head,
                &v[v_offset..v_offset + blocks_per_row],
                weight,
                head_dim,
            );
        }

        let inv_sum = 1.0 / sum_exp;
        for d in (0..head_dim).step_by(4) {
            if d + 4 <= head_dim {
                let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                let scaled = vmulq_f32(out_vec, vdupq_n_f32(inv_sum));
                vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
            } else {
                for dd in d..head_dim {
                    output_head[dd] *= inv_sum;
                }
            }
        }
    }
}

// ============================================================================
// IQ4_NL Attention
// ============================================================================

/// Decode attention for batch=1, IQ4_NL K/V with NEON
/// Strategy: Quantize Q to Q8_0, then use Q8_0 × IQ4_NL fused vec_dot
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn decode_attention_iq4_nl_neon(
    q: &[f32],
    k: &[BlockIQ4NL],
    v: &[BlockIQ4NL],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    let blocks_per_row = head_dim / QK;

    let mut q_q8 = vec![BlockQ8_0 { d: 0, qs: [0i8; QK] }; blocks_per_row];

    for h in 0..num_heads {
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

        quantize_query_to_q8_0(q_head, &mut q_q8);

        let mut max_val = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;

        for d in 0..head_dim {
            output_head[d] = 0.0;
        }

        for s in 0..seq_len {
            let k_offset = (s * num_heads + h) * blocks_per_row;
            let score = vec_dot_iq4_nl_q8_0(
                head_dim,
                &q_q8,
                &k[k_offset..k_offset + blocks_per_row],
            ) * scale;

            let old_max = max_val;
            if score > max_val {
                max_val = score;
            }

            let weight;
            if score > old_max {
                let correction = (old_max - max_val).exp();
                sum_exp *= correction;
                // NEON vector scaling
                let corr_vec = vdupq_n_f32(correction);
                for d in (0..head_dim).step_by(4) {
                    if d + 4 <= head_dim {
                        let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                        let scaled = vmulq_f32(out_vec, corr_vec);
                        vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
                    } else {
                        for dd in d..head_dim {
                            output_head[dd] *= correction;
                        }
                    }
                }
                weight = 1.0;
            } else {
                weight = (score - max_val).exp();
            }

            sum_exp += weight;

            let v_offset = (s * num_heads + h) * blocks_per_row;
            vec_mad_f32_iq4_nl(
                output_head,
                &v[v_offset..v_offset + blocks_per_row],
                weight,
                head_dim,
            );
        }

        let inv_sum = 1.0 / sum_exp;
        for d in (0..head_dim).step_by(4) {
            if d + 4 <= head_dim {
                let out_vec = vld1q_f32(output_head.as_ptr().add(d));
                let scaled = vmulq_f32(out_vec, vdupq_n_f32(inv_sum));
                vst1q_f32(output_head.as_mut_ptr().add(d), scaled);
            } else {
                for dd in d..head_dim {
                    output_head[dd] *= inv_sum;
                }
            }
        }
    }
}

// ============================================================================
// K-quant Attention (scalar implementation for correctness)
// ============================================================================

/// Dequantize Q2_K KV cache block (256 elements) to f32
/// Only returns first head_dim elements (for head_dim < 256)
fn dequantize_q2_k_kv(
    block: &crate::quant::kv_cache::BlockQ2K_KV,
    out: &mut [f32],
) {
    let d = half::f16::from_bits(block.d).to_f32();
    let dmin = half::f16::from_bits(block.dmin).to_f32();

    for sub_block in 0..16 {
        let sc = block.scales[sub_block];
        let dl = d * ((sc & 0x0F) as f32);
        let ml = dmin * ((sc >> 4) as f32);

        // Each sub-block has 16 elements, stored in 4 bytes
        for j in 0..4 {
            let byte = block.qs[sub_block * 4 + j];
            for k in 0..4 {
                let val = ((byte >> (2 * k)) & 0x03) as f32;
                out[sub_block * 16 + j * 4 + k] = dl * val - ml;
            }
        }
    }
}

/// Dequantize Q3_K KV cache block (256 elements) to f32
fn dequantize_q3_k_kv(
    block: &crate::quant::kv_cache::BlockQ3K_KV,
    out: &mut [f32],
) {
    let d = half::f16::from_bits(block.d).to_f32();

    // Q3_K has 8 sub-blocks of 32 elements each
    for sub in 0..8 {
        let sc = block.scales[sub] as f32;
        let start = sub * 32;

        // 32 elements = 8 bytes of qs
        for j in 0..8 {
            let byte = block.qs[sub * 8 + j];
            for k in 0..4 {
                let idx = start + j * 4 + k;
                let low2 = ((byte >> (2 * k)) & 0x03) as f32;
                let high1 = ((block.hmask[idx / 8] >> (idx % 8)) & 1) as f32;
                let val = low2 + high1 * 4.0 - 4.0; // 3-bit signed: [-4, 3]
                out[idx] = d * sc * val;
            }
        }
    }
}

/// Dequantize Q4_K KV cache block (256 elements) to f32
fn dequantize_q4_k_kv(
    block: &crate::quant::kv_cache::BlockQ4K_KV,
    out: &mut [f32],
) {
    let d = half::f16::from_bits(block.d).to_f32();
    let min = half::f16::from_bits(block.dmin).to_f32();

    // Q4_K stores elements 0..127 in low nibbles, elements 128..255 in high nibbles
    // For head_dim <= 128, we use only low nibbles
    for j in 0..128 {
        out[j] = ((block.qs[j] & 0x0F) as f32) * d + min;
    }
    // For head_dim > 128, also dequantize high nibbles
    for j in 0..128 {
        out[128 + j] = (((block.qs[j] >> 4) & 0x0F) as f32) * d + min;
    }
}

/// Dequantize Q5_K KV cache block (256 elements) to f32
fn dequantize_q5_k_kv(
    block: &crate::quant::kv_cache::BlockQ5K_KV,
    out: &mut [f32],
) {
    let d = half::f16::from_bits(block.d).to_f32();
    let min = half::f16::from_bits(block.dmin).to_f32();

    for j in 0..128 {
        let low = (block.qs[j] & 0x0F) as f32;
        let high = ((block.qh[j / 8] >> (j % 8)) & 1) as f32 * 16.0;
        out[j] = (low + high) * d + min;
    }
    for j in 0..128 {
        let low = ((block.qs[j] >> 4) & 0x0F) as f32;
        let high =
            ((block.qh[(128 + j) / 8] >> ((128 + j) % 8)) & 1) as f32 * 16.0;
        out[128 + j] = (low + high) * d + min;
    }
}

/// Dequantize Q6_K KV cache block (256 elements) to f32
/// Only computes first 128 elements (matching Metal shader behavior for head_dim <= 128)
fn dequantize_q6_k_kv(
    block: &crate::quant::kv_cache::BlockQ6K_KV,
    out: &mut [f32],
) {
    let d = half::f16::from_bits(block.d).to_f32();

    // ql[j] = element j (low nibble) | element j+128 (high nibble)
    // We only compute first 128 elements (low nibbles)
    for j in 0..128 {
        let low = (block.ql[j] & 0x0F) as i32;
        let high = ((block.qh[j / 2] >> (4 * (j % 2))) & 0x03) as i32;
        let val = (low + high * 16 - 32) as i8 as f32; // 6-bit signed [-32, 31]
        out[j] = d * val;
    }
}

/// Generic K-quant attention (scalar implementation)
fn decode_attention_kquant_scalar<K, V>(
    q: &[f32],
    k: &[K],
    v: &[V],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
    dequant_k: fn(&K, &mut [f32]),
    dequant_v: fn(&V, &mut [f32]),
) {
    // K-quant uses 256-element blocks
    let mut k_deq = vec![0.0f32; 256];
    let mut v_deq = vec![0.0f32; 256];

    for h in 0..num_heads {
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

        let mut max_val = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;

        for d in 0..head_dim {
            output_head[d] = 0.0;
        }

        for s in 0..seq_len {
            // Each KV row is one K-quant block (256 elements, but we use head_dim)
            let k_block = &k[s * num_heads + h];
            let v_block = &v[s * num_heads + h];

            // Dequantize
            dequant_k(k_block, &mut k_deq);
            dequant_v(v_block, &mut v_deq);

            // Compute score
            let score = vec_dot_f32_scalar(q_head, &k_deq[..head_dim]) * scale;

            // Online softmax
            let old_max = max_val;
            if score > max_val {
                max_val = score;
            }

            let weight;
            if score > old_max {
                let correction = (old_max - max_val).exp();
                sum_exp *= correction;
                for d in 0..head_dim {
                    output_head[d] *= correction;
                }
                weight = 1.0;
            } else {
                weight = (score - max_val).exp();
            }

            sum_exp += weight;

            // Accumulate V
            for d in 0..head_dim {
                output_head[d] += weight * v_deq[d];
            }
        }

        let inv_sum = 1.0 / sum_exp;
        for d in 0..head_dim {
            output_head[d] *= inv_sum;
        }
    }
}

/// Q2_K attention (scalar implementation)
pub fn decode_attention_q2_k(
    q: &[f32],
    k: &[crate::quant::kv_cache::BlockQ2K_KV],
    v: &[crate::quant::kv_cache::BlockQ2K_KV],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    decode_attention_kquant_scalar(
        q,
        k,
        v,
        output,
        num_heads,
        head_dim,
        seq_len,
        scale,
        dequantize_q2_k_kv,
        dequantize_q2_k_kv,
    );
}

/// Q3_K attention (scalar implementation)
pub fn decode_attention_q3_k(
    q: &[f32],
    k: &[crate::quant::kv_cache::BlockQ3K_KV],
    v: &[crate::quant::kv_cache::BlockQ3K_KV],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    decode_attention_kquant_scalar(
        q,
        k,
        v,
        output,
        num_heads,
        head_dim,
        seq_len,
        scale,
        dequantize_q3_k_kv,
        dequantize_q3_k_kv,
    );
}

/// Q4_K attention (scalar implementation)
pub fn decode_attention_q4_k(
    q: &[f32],
    k: &[crate::quant::kv_cache::BlockQ4K_KV],
    v: &[crate::quant::kv_cache::BlockQ4K_KV],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    decode_attention_kquant_scalar(
        q,
        k,
        v,
        output,
        num_heads,
        head_dim,
        seq_len,
        scale,
        dequantize_q4_k_kv,
        dequantize_q4_k_kv,
    );
}

/// Q5_K attention (scalar implementation)
pub fn decode_attention_q5_k(
    q: &[f32],
    k: &[crate::quant::kv_cache::BlockQ5K_KV],
    v: &[crate::quant::kv_cache::BlockQ5K_KV],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    decode_attention_kquant_scalar(
        q,
        k,
        v,
        output,
        num_heads,
        head_dim,
        seq_len,
        scale,
        dequantize_q5_k_kv,
        dequantize_q5_k_kv,
    );
}

/// Q6_K attention (scalar implementation)
pub fn decode_attention_q6_k(
    q: &[f32],
    k: &[crate::quant::kv_cache::BlockQ6K_KV],
    v: &[crate::quant::kv_cache::BlockQ6K_KV],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    decode_attention_kquant_scalar(
        q,
        k,
        v,
        output,
        num_heads,
        head_dim,
        seq_len,
        scale,
        dequantize_q6_k_kv,
        dequantize_q6_k_kv,
    );
}

// ============================================================================
// Helper Functions
// ============================================================================

#[inline]
fn vec_dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[inline]
pub unsafe fn vec_dot_f32_neon(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut sum_vec = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        sum_vec = vfmaq_f32(sum_vec, va, vb);
        i += 4;
    }

    let mut sum = vaddvq_f32(sum_vec);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

// ============================================================================
// Pure Rust FP16 NEON Attention
// ============================================================================

/// Load pair of uint16x8_t (FP16 bits) using ldp instruction (faster than 2x vld1q)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_pair_f16(ptr: *const u16) -> (uint16x8_t, uint16x8_t) {
    let v0: uint16x8_t;
    let v1: uint16x8_t;
    asm!(
        "ldp {0:q}, {1:q}, [{2}]",
        lateout(vreg) v0,
        lateout(vreg) v1,
        in(reg) ptr,
        options(pure, nomem, nostack, preserves_flags)
    );
    (v0, v1)
}

/// FP16 × FP16 dot product using single inline asm block
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn vec_dot_f16_neon(q: &[u16], k: &[u16], n: usize) -> f32 {
    let q_ptr = q.as_ptr();
    let k_ptr = k.as_ptr();
    let np = n & !31;

    // Use a pointer to the result
    let mut result: f32 = 0.0;

    // Store result through a register
    asm!(
        // Initialize 4 FP16 accumulators to zero
        "movi v0.2d, #0",  // sum0
        "movi v1.2d, #0",  // sum1
        "movi v2.2d, #0",  // sum2
        "movi v3.2d, #0",  // sum3
        "mov x8, #0",      // loop counter
        "cmp x2, #32",
        "b.lt 2f",         // skip if n < 32

        // Main loop - process 32 elements per iteration
        "1:",
        // Load 32 q values using ldp
        "ldp q4, q5, [x0, #0]",
        "ldp q6, q7, [x0, #32]",
        // Load 32 k values using ldp
        "ldp q16, q17, [x1, #0]",
        "ldp q18, q19, [x1, #32]",

        // FP16 FMA: sum += q * k
        "fmla v0.8h, v4.8h, v16.8h",
        "fmla v1.8h, v5.8h, v17.8h",
        "fmla v2.8h, v6.8h, v18.8h",
        "fmla v3.8h, v7.8h, v19.8h",

        // Update pointers and counter
        "add x0, x0, #64",
        "add x1, x1, #64",
        "add x8, x8, #32",
        "cmp x8, x2",
        "b.lt 1b",

        // Reduce accumulators
        "2:",
        "fadd v0.8h, v0.8h, v1.8h",
        "fadd v2.8h, v2.8h, v3.8h",
        "fadd v0.8h, v0.8h, v2.8h",

        // Convert FP16 to FP32 and horizontal sum
        "fcvtl v1.4s, v0.4h",
        "fcvtl2 v2.4s, v0.8h",
        "fadd v0.4s, v1.4s, v2.4s",
        // Pairwise add twice to get sum
        "faddp v1.4s, v0.4s, v0.4s",
        "faddp v0.4s, v1.4s, v1.4s",
        // Store result (s0 is the scalar part of v0)
        "str s0, [x3]",

        inout("x0") q_ptr => _,
        inout("x1") k_ptr => _,
        in("x2") np as u64,
        in("x3") &mut result as *mut f32,
        out("v0") _,
        out("v1") _,
        out("v2") _,
        out("v3") _,
        out("v4") _,
        out("v5") _,
        out("v6") _,
        out("v7") _,
        out("v16") _,
        out("v17") _,
        out("v18") _,
        out("v19") _,
        out("x8") _,
        options(nostack)
    );

    // Handle remaining elements
    for j in np..n {
        let q_val = half::f16::from_bits(q[j]).to_f32();
        let k_val = half::f16::from_bits(k[j]).to_f32();
        result += q_val * k_val;
    }
    result
}

/// Convert FP32 to FP16 bits using hardware fcvt instruction
#[cfg(target_arch = "aarch64")]
#[inline]
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

/// FP16 vec_mad with NEON using native FP16 FMA: y += x * v
/// Uses uint16x8_t (stable) to avoid unstable float16x8_t type
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn vec_mad_f16_neon(y: &mut [u16], x: &[u16], v: f32, n: usize) {
    let y_ptr = y.as_mut_ptr();
    let x_ptr = x.as_ptr();

    // Convert scalar v to FP16 - use half crate for scalar conversion
    let v_bits = half::f16::from_f32(v).to_bits();
    // Duplicate to all 8 lanes (no reinterpret needed - just u16)
    let v_f16 = vdupq_n_u16(v_bits);

    let np = n & !7;
    let mut i = 0;

    while i < np {
        // Load 8 FP16 values as u16
        let x8 = vld1q_u16(x_ptr.add(i));
        let y8 = vld1q_u16(y_ptr.add(i));

        // FP16 FMA: y += x * v (inline asm handles reinterpret)
        let result = vfmaq_f16_asm(y8, x8, v_f16);

        // Store back (no reinterpret needed - just u16)
        vst1q_u16(y_ptr.add(i), result);
        i += 8;
    }

    // Handle remaining elements
    for j in np..n {
        let y_val = half::f16::from_bits(y[j]).to_f32();
        let x_val = half::f16::from_bits(x[j]).to_f32();
        y[j] = half::f16::from_f32(y_val + x_val * v).to_bits();
    }
}

/// FP16 vec_scale with NEON using native FP16 MUL: y *= v
/// Uses uint16x8_t (stable) to avoid unstable float16x8_t type
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn vec_scale_f16_neon(y: &mut [u16], v: f32, n: usize) {
    let y_ptr = y.as_mut_ptr();

    // Convert scalar v to FP16 - use half crate for scalar conversion
    let v_bits = half::f16::from_f32(v).to_bits();
    // Duplicate to all 8 lanes (no reinterpret needed - just u16)
    let v_f16 = vdupq_n_u16(v_bits);

    let np = n & !7;
    let mut i = 0;

    while i < np {
        let y8 = vld1q_u16(y_ptr.add(i));
        let result = vmulq_f16_asm(y8, v_f16);
        vst1q_u16(y_ptr.add(i), result);
        i += 8;
    }

    // Handle remaining elements
    for j in np..n {
        let y_val = half::f16::from_bits(y[j]).to_f32();
        y[j] = half::f16::from_f32(y_val * v).to_bits();
    }
}

/// Decode attention for batch=1, FP16 version with NEON
/// All operations inlined - no function call overhead
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn decode_attention_f16_neon(
    q: &[u16],
    k: &[u16],
    v: &[u16],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    // Work buffer for FP16 VKQ accumulator
    let mut vkq: [u16; 256] = [0u16; 256];
    let np = head_dim & !31;

    for h in 0..num_heads {
        let q_head = &q[h * head_dim..];
        let out_head = &mut output[h * head_dim..];

        // Zero accumulator using SIMD
        for d in (0..head_dim).step_by(8) {
            vst1q_u16(vkq.as_mut_ptr().add(d), vdupq_n_u16(0));
        }

        let mut m: f32 = f32::NEG_INFINITY;
        let mut s: f32 = 0.0;

        for pos in 0..seq_len {
            let k_row = &k[(pos * num_heads + h) * head_dim..];
            let v_row = &v[(pos * num_heads + h) * head_dim..];

            // === INLINED vec_dot_f16_neon ===
            let mut q_ptr = q_head.as_ptr();
            let mut k_ptr = k_row.as_ptr();

            q_ptr = q_ptr.add(16);
            k_ptr = k_ptr.add(16);

            let mut sum0: uint16x8_t;
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
            let (sum_f32_0, sum_f32_1) = vcvt_f32_from_f16(sum0);
            let mut score = vaddvq_f32(vaddq_f32(sum_f32_0, sum_f32_1));
            for j in np..head_dim {
                score += half::f16::from_bits(q_head[j]).to_f32()
                    * half::f16::from_bits(k_row[j]).to_f32();
            }
            score *= scale;

            // Online softmax
            if score > m {
                let ms = (m - score).exp();
                s *= ms;
                // === INLINED vec_scale_f16_neon with hardware fcvt ===
                let ms_bits = fp32_to_fp16_bits(ms);
                let ms_vec = vdupq_n_u16(ms_bits);
                for d in (0..head_dim).step_by(8) {
                    let y8 = vld1q_u16(vkq.as_ptr().add(d));
                    vst1q_u16(
                        vkq.as_mut_ptr().add(d),
                        vmulq_f16_asm(y8, ms_vec),
                    );
                }
                m = score;
            }

            let vs = (score - m).exp();
            s += vs;
            // === INLINED vec_mad_f16_neon with hardware fcvt ===
            let vs_bits = fp32_to_fp16_bits(vs);
            let vs_vec = vdupq_n_u16(vs_bits);
            for d in (0..head_dim).step_by(8) {
                let x8 = vld1q_u16(v_row.as_ptr().add(d));
                let y8 = vld1q_u16(vkq.as_ptr().add(d));
                vst1q_u16(
                    vkq.as_mut_ptr().add(d),
                    vfmaq_f16_asm(y8, x8, vs_vec),
                );
            }
        }

        // Normalize and write output
        let inv_s = 1.0 / s;
        for d in 0..head_dim {
            out_head[d] = half::f16::from_bits(vkq[d]).to_f32() * inv_s;
        }
    }
}

/// Decode attention for batch=1, FP32 Q + FP16 K/V (pure Rust intrinsics)
/// Uses NEON FP16 intrinsics - compiler optimizes to ldp/stp
/// Note: Using #[inline(always)] instead of #[target_feature] to enable inlining
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn decode_attention_f32_q_f16_kv_neon(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    // Work buffers
    let mut q_f16: [u16; 256] = [0u16; 256];
    let mut vkq: [u16; 256] = [0u16; 256];

    for h in 0..num_heads {
        let q_head = &q[h * head_dim..];
        let out_head = &mut output[h * head_dim..];

        // Convert Q to FP16 (once per head) using inline asm
        for d in (0..head_dim).step_by(8) {
            let q0 = vld1q_f32(q_head.as_ptr().add(d));
            let q1 = vld1q_f32(q_head.as_ptr().add(d + 4));
            let q_f16_vec = vcvt_f16_from_f32(q0, q1);
            vst1q_u16(q_f16.as_mut_ptr().add(d), q_f16_vec);
        }

        // Zero accumulator using SIMD
        let zero = vdupq_n_u16(0);
        for d in (0..head_dim).step_by(8) {
            vst1q_u16(vkq.as_mut_ptr().add(d), zero);
        }

        let mut m: f32 = f32::NEG_INFINITY;
        let mut s: f32 = 0.0;

        let q_ptr = q_f16.as_ptr();
        let k_base = k.as_ptr();
        let v_base = v.as_ptr();

        for pos in 0..seq_len {
            let k_row = k_base.add((pos * num_heads + h) * head_dim);
            let v_row = v_base.add((pos * num_heads + h) * head_dim);

            // FP16 dot product using inline assembly wrappers
            let mut sum0: uint16x8_t;
            let mut sum1: uint16x8_t;
            let mut sum2: uint16x8_t;
            let mut sum3: uint16x8_t;
            asm!(
                "movi v0.2d, #0\nmovi v1.2d, #0\nmovi v2.2d, #0\nmovi v3.2d, #0",
                lateout("v0") sum0, lateout("v1") sum1, lateout("v2") sum2, lateout("v3") sum3,
                options(nostack)
            );

            // Process 32 FP16 values with 4-way unrolling
            let mut d = 0;
            while d + 32 <= head_dim {
                let (q0, q1) = load_pair_f16(q_ptr.add(d));
                let (k0, k1) = load_pair_f16(k_row.add(d));
                sum0 = vfmaq_f16_asm(sum0, q0, k0);
                sum1 = vfmaq_f16_asm(sum1, q1, k1);

                let (q2, q3) = load_pair_f16(q_ptr.add(d + 16));
                let (k2, k3) = load_pair_f16(k_row.add(d + 16));
                sum2 = vfmaq_f16_asm(sum2, q2, k2);
                sum3 = vfmaq_f16_asm(sum3, q3, k3);

                d += 32;
            }

            // Reduce in FP16 using inline assembly
            sum0 = vaddq_f16_asm(sum0, sum1);
            sum2 = vaddq_f16_asm(sum2, sum3);
            sum0 = vaddq_f16_asm(sum0, sum2);

            // Convert to FP32 and horizontal sum
            let (sum_f32_0, sum_f32_1) = vcvt_f32_from_f16(sum0);
            let sum_vec = vaddq_f32(sum_f32_0, sum_f32_1);
            let mut score = vaddvq_f32(sum_vec);

            // Handle remaining elements
            while d < head_dim {
                let q_val = half::f16::from_bits(q_f16[d]).to_f32();
                let k_val = half::f16::from_bits(k_row.read()).to_f32();
                score += q_val * k_val;
                d += 1;
            }
            score *= scale;

            // Online softmax
            if score > m {
                let ms = (m - score).exp();
                s *= ms;
                m = score;

                // Scale vkq using inline assembly
                let ms_bits = half::f16::from_f32(ms).to_bits();
                let ms_vec: uint16x8_t;
                asm!(
                    "dup {0}.8h, {1:w}",
                    lateout(vreg) ms_vec,
                    in(reg) ms_bits,
                    options(pure, nomem, nostack)
                );

                for d in (0..head_dim).step_by(16) {
                    let (v0, v1) = load_pair_f16(vkq.as_ptr().add(d));
                    vst1q_u16(
                        vkq.as_mut_ptr().add(d),
                        vmulq_f16_asm(v0, ms_vec),
                    );
                    vst1q_u16(
                        vkq.as_mut_ptr().add(d + 8),
                        vmulq_f16_asm(v1, ms_vec),
                    );
                }
            }

            let vs = (score - m).exp();
            s += vs;

            // vec_mad using inline assembly: vkq += v * vs
            let vs_bits = half::f16::from_f32(vs).to_bits();
            let vs_vec: uint16x8_t;
            asm!(
                "dup {0}.8h, {1:w}",
                lateout(vreg) vs_vec,
                in(reg) vs_bits,
                options(pure, nomem, nostack)
            );

            for d in (0..head_dim).step_by(16) {
                let (v0, v1) = load_pair_f16(v_row.add(d));
                let (vkq0, vkq1) = load_pair_f16(vkq.as_ptr().add(d));
                vst1q_u16(
                    vkq.as_mut_ptr().add(d),
                    vfmaq_f16_asm(vkq0, v0, vs_vec),
                );
                vst1q_u16(
                    vkq.as_mut_ptr().add(d + 8),
                    vfmaq_f16_asm(vkq1, v1, vs_vec),
                );
            }
        }

        // Normalize and write output
        let inv_s = 1.0 / s;
        for d in 0..head_dim {
            out_head[d] = half::f16::from_bits(vkq[d]).to_f32() * inv_s;
        }
    }
}

/// Fused FP16 × FP32 vec_dot - avoids intermediate dequantization
/// Takes FP32 Q and FP16 K, computes dot product directly
/// Uses loop unrolling with 4 parallel accumulators for better pipeline utilization
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn vec_dot_f32_f16_neon(
    q_f32: &[f32],
    k_f16: &[u16],
    n: usize,
) -> f32 {
    let ptr_q = q_f32.as_ptr();
    let ptr_k = k_f16.as_ptr();
    let mut i = 0;

    // Use 4 parallel accumulators to reduce pipeline stalls
    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    // Process 16 elements at a time (4 iterations of 4 elements each)
    while i + 16 <= n {
        // Load and convert 4 FP16 to FP32 (iteration 0)
        let k0 = vld1_f16_and_cvt_to_f32(ptr_k.add(i));
        let q0 = vld1q_f32(ptr_q.add(i));
        sum0 = vfmaq_f32(sum0, q0, k0);

        // Load and convert 4 FP16 to FP32 (iteration 1)
        let k1 = vld1_f16_and_cvt_to_f32(ptr_k.add(i + 4));
        let q1 = vld1q_f32(ptr_q.add(i + 4));
        sum1 = vfmaq_f32(sum1, q1, k1);

        // Load and convert 4 FP16 to FP32 (iteration 2)
        let k2 = vld1_f16_and_cvt_to_f32(ptr_k.add(i + 8));
        let q2 = vld1q_f32(ptr_q.add(i + 8));
        sum2 = vfmaq_f32(sum2, q2, k2);

        // Load and convert 4 FP16 to FP32 (iteration 3)
        let k3 = vld1_f16_and_cvt_to_f32(ptr_k.add(i + 12));
        let q3 = vld1q_f32(ptr_q.add(i + 12));
        sum3 = vfmaq_f32(sum3, q3, k3);

        i += 16;
    }

    // Reduce the 4 accumulators
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);

    // Process remaining 4 elements
    if i + 4 <= n {
        let k = vld1_f16_and_cvt_to_f32(ptr_k.add(i));
        let q = vld1q_f32(ptr_q.add(i));
        sum0 = vfmaq_f32(sum0, q, k);
        i += 4;
    }

    let mut sum = vaddvq_f32(sum0);
    // Handle remainder
    while i < n {
        let k_f32 = fp16_to_fp32(k_f16[i]);
        sum += q_f32[i] * k_f32;
        i += 1;
    }
    sum
}

/// Fused FP16 vec_mad - multiply FP16 V by weight and add to FP32 output
/// Uses loop unrolling with 4 parallel operations for better pipeline utilization
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn vec_mad_f32_f16_neon(
    output: &mut [f32],
    v_f16: &[u16],
    weight: f32,
    n: usize,
) {
    let w_vec = vdupq_n_f32(weight);
    let ptr_out = output.as_mut_ptr();
    let ptr_v = v_f16.as_ptr();
    let mut i = 0;

    // Process 16 elements at a time (4 parallel operations)
    while i + 16 <= n {
        // Convert 4 FP16 to FP32 and do FMA (iteration 0)
        let v0 = vld1_f16_and_cvt_to_f32(ptr_v.add(i));
        let out0 = vld1q_f32(ptr_out.add(i));
        vst1q_f32(ptr_out.add(i), vfmaq_f32(out0, v0, w_vec));

        // Convert 4 FP16 to FP32 and do FMA (iteration 1)
        let v1 = vld1_f16_and_cvt_to_f32(ptr_v.add(i + 4));
        let out1 = vld1q_f32(ptr_out.add(i + 4));
        vst1q_f32(ptr_out.add(i + 4), vfmaq_f32(out1, v1, w_vec));

        // Convert 4 FP16 to FP32 and do FMA (iteration 2)
        let v2 = vld1_f16_and_cvt_to_f32(ptr_v.add(i + 8));
        let out2 = vld1q_f32(ptr_out.add(i + 8));
        vst1q_f32(ptr_out.add(i + 8), vfmaq_f32(out2, v2, w_vec));

        // Convert 4 FP16 to FP32 and do FMA (iteration 3)
        let v3 = vld1_f16_and_cvt_to_f32(ptr_v.add(i + 12));
        let out3 = vld1q_f32(ptr_out.add(i + 12));
        vst1q_f32(ptr_out.add(i + 12), vfmaq_f32(out3, v3, w_vec));

        i += 16;
    }

    // Process remaining 4 elements
    if i + 4 <= n {
        let v = vld1_f16_and_cvt_to_f32(ptr_v.add(i));
        let out = vld1q_f32(ptr_out.add(i));
        vst1q_f32(ptr_out.add(i), vfmaq_f32(out, v, w_vec));
        i += 4;
    }

    // Handle remainder
    while i < n {
        let v_f32 = fp16_to_fp32(v_f16[i]);
        output[i] += weight * v_f32;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn fp16_to_fp32(bits: u16) -> f32 {
    // Use inline asm to avoid unstable float16x4_t type
    let result: f32;
    asm!(
        "dup v0.4h, {1:w}",
        "fcvtl v0.4s, v0.4h",
        "fmov {0}, s0",
        out(reg) result,
        in(reg) bits,
        out("v0") _,
        options(pure, nomem, nostack)
    );
    result
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn bf16_to_fp32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[inline]
pub unsafe fn dequantize_row_bf16_neon_inline(x: &[u16], y: &mut [f32]) {
    let n = x.len();
    let n4 = n / 4;

    for i in 0..n4 {
        let offset = i * 4;
        let x_u16 = vld1_u16(x.as_ptr().add(offset));
        let x_u32 = vmovl_u16(x_u16);
        let y_vec = vreinterpretq_f32_u32(vshlq_n_u32(x_u32, 16));
        vst1q_f32(y.as_mut_ptr().add(offset), y_vec);
    }

    for i in (n4 * 4)..n {
        y[i] = bf16_to_fp32(x[i]);
    }
}

// Fused vec_dot and vec_mad functions are provided by attention_fused module
#[cfg(target_arch = "aarch64")]
use super::attention_fused::{
    quantize_query_to_q8_0, quantize_query_to_q8_1, vec_dot_iq4_nl_q8_0,
    vec_dot_q4_0_q8_0, vec_dot_q5_0_q8_0, vec_mad_f32_iq4_nl, vec_mad_f32_q4_0,
    vec_mad_f32_q4_1, vec_mad_f32_q5_0, vec_mad_f32_q5_1, vec_mad_f32_q8_0,
};

// ============================================================================
// Scalar fallbacks for non-aarch64 platforms
// ============================================================================

/// Scalar FP32 vec_dot
#[cfg(not(target_arch = "aarch64"))]
fn vec_dot_f32_scalar(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar FP16 to FP32 conversion
fn fp16_to_fp32_scalar(bits: u16) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { crate::ops::arm::neon::fp16_to_fp32(bits) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        half::f16::from_bits(bits).to_f32()
    }
}

/// Scalar BF16 to FP32 conversion
fn bf16_to_fp32_scalar(bits: u16) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { crate::ops::arm::neon::bf16_to_fp32(bits) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        f32::from_bits((bits as u32) << 16)
    }
}

/// Scalar FP16 vec_dot
fn vec_dot_f16_scalar(q: &[u16], k: &[u16], n: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { crate::ops::arm::neon::vec_dot_f16(q, k, n) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += fp16_to_fp32_scalar(q[i]) * fp16_to_fp32_scalar(k[i]);
        }
        sum
    }
}

/// Scalar FP32 query, FP16 KV vec_dot
fn vec_dot_f32_f16_scalar(q: &[f32], k: &[u16], n: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { crate::ops::arm::neon::vec_dot_f32_f16(q, k, n) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += q[i] * fp16_to_fp32_scalar(k[i]);
        }
        sum
    }
}

/// Scalar FP16 vec_mad: y += x * v
fn vec_mad_f16_scalar(y: &mut [u16], x: &[u16], v: f32, n: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { crate::ops::arm::neon::vec_mad_f16(y, x, v, n) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            let y_val = fp16_to_fp32_scalar(y[i]);
            let x_val = fp16_to_fp32_scalar(x[i]);
            y[i] = half::f16::from_f32(y_val + x_val * v).to_bits();
        }
    }
}

/// Scalar FP32 output, FP16 V vec_mad
fn vec_mad_f32_f16_scalar(y: &mut [f32], v: &[u16], weight: f32, n: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { crate::ops::arm::neon::vec_mad_f32_f16(y, v, weight, n) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            y[i] += weight * fp16_to_fp32_scalar(v[i]);
        }
    }
}

/// Scalar FP16 scale
fn vec_scale_f16_scalar(y: &mut [u16], v: f32, n: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { crate::ops::arm::neon::vec_scale_f16(y, v, n) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            let y_val = fp16_to_fp32_scalar(y[i]);
            y[i] = half::f16::from_f32(y_val * v).to_bits();
        }
    }
}

/// Scalar dequantize BF16 row
fn dequantize_row_bf16_scalar(x: &[u16], y: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { crate::ops::arm::neon::dequantize_row_bf16(x, y) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..x.len() {
            y[i] = bf16_to_fp32_scalar(x[i]);
        }
    }
}

/// Scalar dequantize FP16 row
fn dequantize_row_f16_scalar(x: &[u16], y: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { crate::ops::arm::neon::dequantize_row_f16(x, y) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..x.len() {
            y[i] = fp16_to_fp32_scalar(x[i]);
        }
    }
}

// ============================================================================
// Dispatch functions for standard quantization formats
// ============================================================================
// Dispatch functions for standard quantization formats
// ============================================================================

/// BF16 attention
pub fn decode_attention_bf16(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            decode_attention_bf16_neon(
                q, k, v, output, num_heads, head_dim, seq_len, scale,
            )
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut k_f32 = vec![0.0f32; head_dim];
        let mut v_f32 = vec![0.0f32; head_dim];

        for h in 0..num_heads {
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

            let mut max_val = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;

            output_head.fill(0.0);

            for s in 0..seq_len {
                let k_offset = (s * num_heads + h) * head_dim;
                dequantize_row_bf16_scalar(
                    &k[k_offset..k_offset + head_dim],
                    &mut k_f32,
                );

                let score =
                    vec_dot_f32_scalar(q_head, &k_f32, head_dim) * scale;

                let old_max = max_val;
                if score > max_val {
                    max_val = score;
                }

                let weight;
                if score > old_max {
                    let correction = (old_max - max_val).exp();
                    sum_exp *= correction;
                    for d in 0..head_dim {
                        output_head[d] *= correction;
                    }
                    weight = 1.0;
                } else {
                    weight = (score - max_val).exp();
                }

                sum_exp += weight;

                dequantize_row_bf16_scalar(
                    &v[k_offset..k_offset + head_dim],
                    &mut v_f32,
                );
                for d in 0..head_dim {
                    output_head[d] += weight * v_f32[d];
                }
            }

            let inv_sum = 1.0 / sum_exp;
            for d in 0..head_dim {
                output_head[d] *= inv_sum;
            }
        }
    }
}

/// FP16 attention
/// Note: This accepts FP32 query and uses FP16 KV.
/// For full FP16 query+KV, use decode_attention_f16_neon directly.
pub fn decode_attention_f16(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    // Use f32 query + f16 KV implementation
    decode_attention_f32_q_f16_kv(
        q, k, v, output, num_heads, head_dim, seq_len, scale,
    );
}

/// FP32 query with FP16 KV attention
pub fn decode_attention_f32_q_f16_kv(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            decode_attention_f32_q_f16_kv_neon(
                q, k, v, output, num_heads, head_dim, seq_len, scale,
            )
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for h in 0..num_heads {
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

            let mut max_val = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;

            output_head.fill(0.0);

            for s in 0..seq_len {
                let k_offset = (s * num_heads + h) * head_dim;
                let k_row = &k[k_offset..k_offset + head_dim];

                let score =
                    vec_dot_f32_f16_scalar(q_head, k_row, head_dim) * scale;

                let old_max = max_val;
                if score > max_val {
                    max_val = score;
                }

                let weight;
                if score > old_max {
                    let correction = (old_max - max_val).exp();
                    sum_exp *= correction;
                    for d in 0..head_dim {
                        output_head[d] *= correction;
                    }
                    weight = 1.0;
                } else {
                    weight = (score - max_val).exp();
                }

                sum_exp += weight;

                let v_row = &v[k_offset..k_offset + head_dim];
                for d in 0..head_dim {
                    output_head[d] += weight * fp16_to_fp32_scalar(v_row[d]);
                }
            }

            let inv_sum = 1.0 / sum_exp;
            for d in 0..head_dim {
                output_head[d] *= inv_sum;
            }
        }
    }
}

/// Q8_0 attention
pub fn decode_attention_q8_0(
    q: &[f32],
    k: &[BlockQ8_0],
    v: &[BlockQ8_0],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            decode_attention_q8_0_neon(
                q, k, v, output, num_heads, head_dim, seq_len, scale,
            )
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let blocks_per_row = (head_dim + QK - 1) / QK;
        let mut q_q8 = vec![BlockQ8_0::default(); blocks_per_row];

        for h in 0..num_heads {
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

            // Quantize query to Q8_0
            crate::quant::kv_cache::quant::quantize_row_q8_0_scalar(
                q_head, &mut q_q8,
            );

            let mut max_val = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;

            output_head.fill(0.0);

            for s in 0..seq_len {
                let kv_offset = (s * num_heads + h) * blocks_per_row;
                let k_blocks = &k[kv_offset..kv_offset + blocks_per_row];

                let score = crate::quant::vec_dot::vec_dot_q8_0_q8_0(
                    head_dim, &q_q8, k_blocks,
                ) * scale;

                let old_max = max_val;
                if score > max_val {
                    max_val = score;
                }

                let weight;
                if score > old_max {
                    let correction = (old_max - max_val).exp();
                    sum_exp *= correction;
                    for d in 0..head_dim {
                        output_head[d] *= correction;
                    }
                    weight = 1.0;
                } else {
                    weight = (score - max_val).exp();
                }

                sum_exp += weight;

                // Accumulate V
                let v_blocks = &v[kv_offset..kv_offset + blocks_per_row];
                for (bi, block) in v_blocks.iter().enumerate() {
                    let block_scale = fp16_to_fp32_scalar(block.d);
                    for j in 0..QK {
                        let idx = bi * QK + j;
                        if idx < head_dim {
                            output_head[idx] +=
                                weight * block_scale * block.qs[j] as f32;
                        }
                    }
                }
            }

            let inv_sum = 1.0 / sum_exp;
            for d in 0..head_dim {
                output_head[d] *= inv_sum;
            }
        }
    }
}

/// Q4_0 attention
pub fn decode_attention_q4_0(
    q: &[f32],
    k: &[BlockQ4_0],
    v: &[BlockQ4_0],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            decode_attention_q4_0_neon(
                q, k, v, output, num_heads, head_dim, seq_len, scale,
            )
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let blocks_per_row = (head_dim + QK - 1) / QK;
        let mut q_q8 = vec![BlockQ8_0::default(); blocks_per_row];

        for h in 0..num_heads {
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

            crate::quant::kv_cache::quant::quantize_row_q8_0_scalar(
                q_head, &mut q_q8,
            );

            let mut max_val = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;

            output_head.fill(0.0);

            for s in 0..seq_len {
                let kv_offset = (s * num_heads + h) * blocks_per_row;
                let k_blocks = &k[kv_offset..kv_offset + blocks_per_row];

                let score = crate::quant::vec_dot::vec_dot_q4_0_q8_0(
                    head_dim, &q_q8, k_blocks,
                ) * scale;

                let old_max = max_val;
                if score > max_val {
                    max_val = score;
                }

                let weight;
                if score > old_max {
                    let correction = (old_max - max_val).exp();
                    sum_exp *= correction;
                    for d in 0..head_dim {
                        output_head[d] *= correction;
                    }
                    weight = 1.0;
                } else {
                    weight = (score - max_val).exp();
                }

                sum_exp += weight;

                let v_blocks = &v[kv_offset..kv_offset + blocks_per_row];
                for (bi, block) in v_blocks.iter().enumerate() {
                    let block_scale = fp16_to_fp32_scalar(block.d);
                    for j in 0..QK / 2 {
                        let idx_lo = bi * QK + j;
                        let idx_hi = bi * QK + j + QK / 2;
                        let packed = block.qs[j];
                        let lo = ((packed & 0x0F) as i8 as i32 - 8) as f32;
                        let hi =
                            (((packed >> 4) & 0x0F) as i8 as i32 - 8) as f32;
                        if idx_lo < head_dim {
                            output_head[idx_lo] += weight * block_scale * lo;
                        }
                        if idx_hi < head_dim {
                            output_head[idx_hi] += weight * block_scale * hi;
                        }
                    }
                }
            }

            let inv_sum = 1.0 / sum_exp;
            for d in 0..head_dim {
                output_head[d] *= inv_sum;
            }
        }
    }
}

/// Q4_1 attention
pub fn decode_attention_q4_1(
    q: &[f32],
    k: &[BlockQ4_1],
    v: &[BlockQ4_1],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            decode_attention_q4_1_neon(
                q, k, v, output, num_heads, head_dim, seq_len, scale,
            )
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let blocks_per_row = (head_dim + QK - 1) / QK;
        let mut q_q8 = vec![BlockQ8_0::default(); blocks_per_row];

        for h in 0..num_heads {
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

            crate::quant::kv_cache::quant::quantize_row_q8_0_scalar(
                q_head, &mut q_q8,
            );

            let mut max_val = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;

            output_head.fill(0.0);

            for s in 0..seq_len {
                let kv_offset = (s * num_heads + h) * blocks_per_row;
                let k_blocks = &k[kv_offset..kv_offset + blocks_per_row];

                let score = crate::quant::vec_dot::vec_dot_q4_1_q8_0(
                    head_dim, &q_q8, k_blocks,
                ) * scale;

                let old_max = max_val;
                if score > max_val {
                    max_val = score;
                }

                let weight;
                if score > old_max {
                    let correction = (old_max - max_val).exp();
                    sum_exp *= correction;
                    for d in 0..head_dim {
                        output_head[d] *= correction;
                    }
                    weight = 1.0;
                } else {
                    weight = (score - max_val).exp();
                }

                sum_exp += weight;

                let v_blocks = &v[kv_offset..kv_offset + blocks_per_row];
                for (bi, block) in v_blocks.iter().enumerate() {
                    let bs = fp16_to_fp32_scalar(block.d);
                    let bm = fp16_to_fp32_scalar(block.m);
                    for j in 0..QK / 2 {
                        let idx_lo = bi * QK + j;
                        let idx_hi = bi * QK + j + QK / 2;
                        let packed = block.qs[j];
                        let lo = (packed & 0x0F) as f32;
                        let hi = ((packed >> 4) & 0x0F) as f32;
                        if idx_lo < head_dim {
                            output_head[idx_lo] += weight * (bs * lo + bm);
                        }
                        if idx_hi < head_dim {
                            output_head[idx_hi] += weight * (bs * hi + bm);
                        }
                    }
                }
            }

            let inv_sum = 1.0 / sum_exp;
            for d in 0..head_dim {
                output_head[d] *= inv_sum;
            }
        }
    }
}

/// Q5_0 attention
pub fn decode_attention_q5_0(
    q: &[f32],
    k: &[BlockQ5_0],
    v: &[BlockQ5_0],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            decode_attention_q5_0_neon(
                q, k, v, output, num_heads, head_dim, seq_len, scale,
            )
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let blocks_per_row = (head_dim + QK - 1) / QK;
        let mut q_q8 = vec![BlockQ8_0::default(); blocks_per_row];

        for h in 0..num_heads {
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

            crate::quant::kv_cache::quant::quantize_row_q8_0_scalar(
                q_head, &mut q_q8,
            );

            let mut max_val = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;

            output_head.fill(0.0);

            for s in 0..seq_len {
                let kv_offset = (s * num_heads + h) * blocks_per_row;
                let k_blocks = &k[kv_offset..kv_offset + blocks_per_row];

                let score = crate::quant::vec_dot::vec_dot_q5_0_q8_0(
                    head_dim, &q_q8, k_blocks,
                ) * scale;

                let old_max = max_val;
                if score > max_val {
                    max_val = score;
                }

                let weight;
                if score > old_max {
                    let correction = (old_max - max_val).exp();
                    sum_exp *= correction;
                    for d in 0..head_dim {
                        output_head[d] *= correction;
                    }
                    weight = 1.0;
                } else {
                    weight = (score - max_val).exp();
                }

                sum_exp += weight;

                let v_blocks = &v[kv_offset..kv_offset + blocks_per_row];
                for (bi, block) in v_blocks.iter().enumerate() {
                    let bs = fp16_to_fp32_scalar(block.d);
                    let qh = u32::from_le_bytes(block.qh);
                    for j in 0..QK / 2 {
                        let idx_lo = bi * QK + j;
                        let idx_hi = bi * QK + j + QK / 2;
                        let packed = block.qs[j];
                        let lo = ((packed & 0x0F) as i32
                            + ((qh >> j) & 1) as i32 * 16
                            - 16) as f32;
                        let hi = (((packed >> 4) & 0x0F) as i32
                            + ((qh >> (j + 16)) & 1) as i32 * 16
                            - 16) as f32;
                        if idx_lo < head_dim {
                            output_head[idx_lo] += weight * bs * lo;
                        }
                        if idx_hi < head_dim {
                            output_head[idx_hi] += weight * bs * hi;
                        }
                    }
                }
            }

            let inv_sum = 1.0 / sum_exp;
            for d in 0..head_dim {
                output_head[d] *= inv_sum;
            }
        }
    }
}

/// Q5_1 attention
pub fn decode_attention_q5_1(
    q: &[f32],
    k: &[BlockQ5_1],
    v: &[BlockQ5_1],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            decode_attention_q5_1_neon(
                q, k, v, output, num_heads, head_dim, seq_len, scale,
            )
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let blocks_per_row = (head_dim + QK - 1) / QK;
        let mut q_q8 = vec![BlockQ8_0::default(); blocks_per_row];

        for h in 0..num_heads {
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

            crate::quant::kv_cache::quant::quantize_row_q8_0_scalar(
                q_head, &mut q_q8,
            );

            let mut max_val = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;

            output_head.fill(0.0);

            for s in 0..seq_len {
                let kv_offset = (s * num_heads + h) * blocks_per_row;
                let k_blocks = &k[kv_offset..kv_offset + blocks_per_row];

                let score = crate::quant::vec_dot::vec_dot_q5_1_q8_0(
                    head_dim, &q_q8, k_blocks,
                ) * scale;

                let old_max = max_val;
                if score > max_val {
                    max_val = score;
                }

                let weight;
                if score > old_max {
                    let correction = (old_max - max_val).exp();
                    sum_exp *= correction;
                    for d in 0..head_dim {
                        output_head[d] *= correction;
                    }
                    weight = 1.0;
                } else {
                    weight = (score - max_val).exp();
                }

                sum_exp += weight;

                let v_blocks = &v[kv_offset..kv_offset + blocks_per_row];
                for (bi, block) in v_blocks.iter().enumerate() {
                    let bs = fp16_to_fp32_scalar(block.d);
                    let bm = fp16_to_fp32_scalar(block.m);
                    let qh = u32::from_le_bytes(block.qh);
                    for j in 0..QK / 2 {
                        let idx_lo = bi * QK + j;
                        let idx_hi = bi * QK + j + QK / 2;
                        let packed = block.qs[j];
                        let lo = ((packed & 0x0F) as f32
                            + ((qh >> j) & 1) as f32 * 16.0);
                        let hi = (((packed >> 4) & 0x0F) as f32
                            + ((qh >> (j + 16)) & 1) as f32 * 16.0);
                        if idx_lo < head_dim {
                            output_head[idx_lo] += weight * (bs * lo + bm);
                        }
                        if idx_hi < head_dim {
                            output_head[idx_hi] += weight * (bs * hi + bm);
                        }
                    }
                }
            }

            let inv_sum = 1.0 / sum_exp;
            for d in 0..head_dim {
                output_head[d] *= inv_sum;
            }
        }
    }
}

/// IQ4_NL attention
pub fn decode_attention_iq4_nl(
    q: &[f32],
    k: &[BlockIQ4NL],
    v: &[BlockIQ4NL],
    output: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            decode_attention_iq4_nl_neon(
                q, k, v, output, num_heads, head_dim, seq_len, scale,
            )
        };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        use crate::quant::kv_cache::KVALUES_IQ4NL;

        let blocks_per_row = (head_dim + QK - 1) / QK;
        let mut q_q8 = vec![BlockQ8_0::default(); blocks_per_row];

        for h in 0..num_heads {
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let output_head = &mut output[h * head_dim..(h + 1) * head_dim];

            crate::quant::kv_cache::quant::quantize_row_q8_0_scalar(
                q_head, &mut q_q8,
            );

            let mut max_val = f32::NEG_INFINITY;
            let mut sum_exp = 0.0f32;

            output_head.fill(0.0);

            for s in 0..seq_len {
                let kv_offset = (s * num_heads + h) * blocks_per_row;
                let k_blocks = &k[kv_offset..kv_offset + blocks_per_row];

                let score = crate::quant::vec_dot::vec_dot_iq4_nl_q8_0(
                    head_dim, &q_q8, k_blocks,
                ) * scale;

                let old_max = max_val;
                if score > max_val {
                    max_val = score;
                }

                let weight;
                if score > old_max {
                    let correction = (old_max - max_val).exp();
                    sum_exp *= correction;
                    for d in 0..head_dim {
                        output_head[d] *= correction;
                    }
                    weight = 1.0;
                } else {
                    weight = (score - max_val).exp();
                }

                sum_exp += weight;

                let v_blocks = &v[kv_offset..kv_offset + blocks_per_row];
                for (bi, block) in v_blocks.iter().enumerate() {
                    let bs = fp16_to_fp32_scalar(block.d);
                    for j in 0..QK / 2 {
                        let idx_lo = bi * QK + j;
                        let idx_hi = bi * QK + j + QK / 2;
                        let packed = block.qs[j];
                        let lo = KVALUES_IQ4NL[(packed & 0x0F) as usize] as f32;
                        let hi = KVALUES_IQ4NL[((packed >> 4) & 0x0F) as usize]
                            as f32;
                        if idx_lo < head_dim {
                            output_head[idx_lo] += weight * bs * lo;
                        }
                        if idx_hi < head_dim {
                            output_head[idx_hi] += weight * bs * hi;
                        }
                    }
                }
            }

            let inv_sum = 1.0 / sum_exp;
            for d in 0..head_dim {
                output_head[d] *= inv_sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_attention_f32_basic() {
        let num_heads = 2;
        let head_dim = 4;
        let seq_len = 3;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = vec![1.0f32; num_heads * head_dim];
        let k = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.0, 2.0,
            2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0,
        ];
        let v = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5,
        ];

        let mut output = vec![0.0f32; num_heads * head_dim];

        decode_attention_f32(
            &q,
            &k,
            &v,
            &mut output,
            num_heads,
            head_dim,
            seq_len,
            scale,
        );

        let sum: f32 = output.iter().sum();
        assert!(sum > 0.0, "Output should have positive values");
    }
}
