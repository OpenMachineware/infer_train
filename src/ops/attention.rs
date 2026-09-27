// Attention operator implementation with fused KV dequantization
// Based on llama.cpp's flash attention mechanism

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::quant::kv_cache::{
    BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockQ8_1, BlockIQ4NL, QK,
    dequant::dequantize_row_f16_neon,
};
use crate::ops::softmax::softmax_f32_dispatch;

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
            let k_row = &k[(s * num_heads + h) * head_dim..(s * num_heads + h + 1) * head_dim];
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
            dequantize_row_bf16_neon_inline(&k[k_offset..k_offset + head_dim], &mut k_f32);

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
                let q_ptr = q_q8.as_ptr() as *const crate::quant::types::BlockQ8_0;
                let q_slice = std::slice::from_raw_parts(q_ptr, blocks_per_row);
                let k_ptr = k.as_ptr().add(k_offset) as *const crate::quant::types::BlockQ8_0;
                let k_slice = std::slice::from_raw_parts(k_ptr, blocks_per_row);
                crate::quant::vec_dot::arm::q_0_1::vec_dot_q8_0_q8_0_neon(head_dim, q_slice, k_slice)
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
                    vst1q_f32(output_head.as_mut_ptr().add(d), vmulq_f32(out_vec, corr_vec));
                }
                weight = 1.0;
            } else {
                weight = (score - max_val).exp();
            }

            sum_exp += weight;

            // Accumulate V with NEON vec_mad
            let v_offset = (s * num_heads + h) * blocks_per_row;
            vec_mad_f32_q8_0_neon(output_head, &v[v_offset..v_offset + blocks_per_row], weight, head_dim);
        }

        let inv_sum = 1.0 / sum_exp;
        for d in (0..head_dim).step_by(4) {
            let out_vec = vld1q_f32(output_head.as_ptr().add(d));
            vst1q_f32(output_head.as_mut_ptr().add(d), vmulq_f32(out_vec, vdupq_n_f32(inv_sum)));
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
            let score = vec_dot_q4_0_q8_0(head_dim, &q_q8, &k[k_offset..k_offset + blocks_per_row]) * scale;

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
            vec_mad_f32_q4_0_neon(output_head, &v[v_offset..v_offset + blocks_per_row], weight, head_dim);
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
    let mut q_q8 = vec![BlockQ8_1 { d: 0, s: 0, qs: [0i8; QK] }; blocks_per_row];

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
                let k_ptr = k.as_ptr().add(k_offset) as *const crate::quant::types::BlockQ4_1;
                let k_slice = std::slice::from_raw_parts(k_ptr, blocks_per_row);
                let q_ptr = q_q8.as_ptr() as *const crate::quant::types::BlockQ8_1;
                let q_slice = std::slice::from_raw_parts(q_ptr, blocks_per_row);
                crate::quant::vec_dot::arm::q_0_1::vec_dot_q4_1_q8_1_neon(head_dim, k_slice, q_slice)
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
            vec_mad_f32_q4_1_neon(output_head, &v[v_offset..v_offset + blocks_per_row], weight, head_dim);
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
            let score = vec_dot_q5_0_q8_0(head_dim, &q_q8, &k[k_offset..k_offset + blocks_per_row]) * scale;

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
            vec_mad_f32_q5_0_neon(output_head, &v[v_offset..v_offset + blocks_per_row], weight, head_dim);
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
    let mut q_q8 = vec![BlockQ8_1 { d: 0, s: 0, qs: [0i8; QK] }; blocks_per_row];

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
                let k_ptr = k.as_ptr().add(k_offset) as *const crate::quant::types::BlockQ5_1;
                let k_slice = std::slice::from_raw_parts(k_ptr, blocks_per_row);
                let q_ptr = q_q8.as_ptr() as *const crate::quant::types::BlockQ8_1;
                let q_slice = std::slice::from_raw_parts(q_ptr, blocks_per_row);
                crate::quant::vec_dot::arm::q_0_1::vec_dot_q5_1_q8_1_neon(head_dim, k_slice, q_slice)
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
            vec_mad_f32_q5_1_neon(output_head, &v[v_offset..v_offset + blocks_per_row], weight, head_dim);
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
            let score = vec_dot_iq4_nl_q8_0(head_dim, &q_q8, &k[k_offset..k_offset + blocks_per_row]) * scale;

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
            vec_mad_f32_iq4_nl_neon(output_head, &v[v_offset..v_offset + blocks_per_row], weight, head_dim);
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
// Native FP16 SIMD via C library
// ============================================================================

#[cfg(target_arch = "aarch64")]
unsafe extern "C" {
    /// Complete FP16 attention kernel (C implementation)
    fn decode_attention_f16_native(
        q: *const u16,
        k: *const u16,
        v: *const u16,
        output: *mut f32,
        num_heads: i32,
        head_dim: i32,
        seq_len: i32,
        scale: f32,
        vkq16: *mut u8,  // Pre-allocated work buffer
    );

    /// Native FP16 × FP16 dot product (C implementation)
    /// Uses `float16x8_t` and `vfmaq_f16` for maximum performance
    fn vec_dot_f16_native(q: *const u16, k: *const u16, n: i32) -> f32;

    /// Native FP16 vec_mad (C implementation)
    fn vec_mad_f16_f32_native(output: *mut f32, v: *const u16, weight: f32, n: i32);
}

/// Decode attention for batch=1, FP16 version with NEON
/// Uses native C implementation for maximum performance
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
    // Use stack allocation for work buffer (512 bytes is small and cache-friendly)
    // C kernel will zero via memset before use
    let mut vkq16: [u8; 512] = [0u8; 512];

    // Call complete C kernel
    decode_attention_f16_native(
        q.as_ptr(),
        k.as_ptr(),
        v.as_ptr(),
        output.as_mut_ptr(),
        num_heads as i32,
        head_dim as i32,
        seq_len as i32,
        scale,
        vkq16.as_mut_ptr(),
    );
}

/// Fused FP16 × FP32 vec_dot - avoids intermediate dequantization
/// Takes FP32 Q and FP16 K, computes dot product directly
/// Uses loop unrolling with 4 parallel accumulators for better pipeline utilization
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn vec_dot_f32_f16_neon(q_f32: &[f32], k_f16: &[u16], n: usize) -> f32 {
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
        let k0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(ptr_k.add(i))));
        let q0 = vld1q_f32(ptr_q.add(i));
        sum0 = vfmaq_f32(sum0, q0, k0);

        // Load and convert 4 FP16 to FP32 (iteration 1)
        let k1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(ptr_k.add(i + 4))));
        let q1 = vld1q_f32(ptr_q.add(i + 4));
        sum1 = vfmaq_f32(sum1, q1, k1);

        // Load and convert 4 FP16 to FP32 (iteration 2)
        let k2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(ptr_k.add(i + 8))));
        let q2 = vld1q_f32(ptr_q.add(i + 8));
        sum2 = vfmaq_f32(sum2, q2, k2);

        // Load and convert 4 FP16 to FP32 (iteration 3)
        let k3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(ptr_k.add(i + 12))));
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
        let k = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(ptr_k.add(i))));
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
pub unsafe fn vec_mad_f32_f16_neon(output: &mut [f32], v_f16: &[u16], weight: f32, n: usize) {
    let w_vec = vdupq_n_f32(weight);
    let ptr_out = output.as_mut_ptr();
    let ptr_v = v_f16.as_ptr();
    let mut i = 0;

    // Process 16 elements at a time (4 parallel operations)
    while i + 16 <= n {
        // Convert 4 FP16 to FP32 and do FMA (iteration 0)
        let v0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(ptr_v.add(i))));
        let out0 = vld1q_f32(ptr_out.add(i));
        vst1q_f32(ptr_out.add(i), vfmaq_f32(out0, v0, w_vec));

        // Convert 4 FP16 to FP32 and do FMA (iteration 1)
        let v1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(ptr_v.add(i + 4))));
        let out1 = vld1q_f32(ptr_out.add(i + 4));
        vst1q_f32(ptr_out.add(i + 4), vfmaq_f32(out1, v1, w_vec));

        // Convert 4 FP16 to FP32 and do FMA (iteration 2)
        let v2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(ptr_v.add(i + 8))));
        let out2 = vld1q_f32(ptr_out.add(i + 8));
        vst1q_f32(ptr_out.add(i + 8), vfmaq_f32(out2, v2, w_vec));

        // Convert 4 FP16 to FP32 and do FMA (iteration 3)
        let v3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(ptr_v.add(i + 12))));
        let out3 = vld1q_f32(ptr_out.add(i + 12));
        vst1q_f32(ptr_out.add(i + 12), vfmaq_f32(out3, v3, w_vec));

        i += 16;
    }

    // Process remaining 4 elements
    if i + 4 <= n {
        let v = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(ptr_v.add(i))));
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
    let f16 = vreinterpret_f16_u16(vdup_n_u16(bits));
    let f32_vec = vcvt_f32_f16(f16);
    vgetq_lane_f32(f32_vec, 0)
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
use super::attention_fused::{
    quantize_query_to_q8_0,
    quantize_query_to_q8_1,
    fp16_to_fp32_scalar,
    vec_dot_q8_0_q8_0,
    vec_dot_q4_0_q8_0,
    vec_dot_q4_1_q8_1,
    vec_dot_q5_0_q8_0,
    vec_dot_q5_1_q8_1,
    vec_dot_iq4_nl_q8_0,
    vec_mad_f32_q8_0_neon,
    vec_mad_f32_q4_0_neon,
    vec_mad_f32_q4_1_neon,
    vec_mad_f32_q5_0_neon,
    vec_mad_f32_q5_1_neon,
    vec_mad_f32_iq4_nl_neon,
};

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
            1.0, 2.0, 3.0, 4.0,
            2.0, 3.0, 4.0, 5.0,
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
            0.5, 0.5, 0.5, 0.5,
            1.0, 1.0, 1.0, 1.0,
        ];
        let v = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            0.5, 0.5, 0.5, 0.5,
        ];

        let mut output = vec![0.0f32; num_heads * head_dim];

        decode_attention_f32(&q, &k, &v, &mut output, num_heads, head_dim, seq_len, scale);

        let sum: f32 = output.iter().sum();
        assert!(sum > 0.0, "Output should have positive values");
    }
}
