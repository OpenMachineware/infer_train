// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// NEON SIMD implementations for ARM64

pub mod attention;
pub mod attention_fused;
pub mod binary;
pub mod conv;
pub mod dup;
pub mod gelu;
pub mod moe;
pub mod rms_norm;
pub mod sampling;
pub mod silu;
pub mod softmax;
pub mod tensor;

pub use attention::{
    bf16_to_fp32, dequantize_row_bf16, dequantize_row_f16, fp16_to_fp32,
    fp32_to_fp16_bits, vaddq_f16_asm, vaddvq_f32_inline, vcvt_f16_from_f32,
    vcvt_f32_from_f16, vec_dot_f16, vec_dot_f32, vec_dot_f32_f16, vec_mad_f16,
    vec_mad_f32_f16, vec_scale_f16, vfmaq_f16_asm, vld1_f16_and_cvt_to_f32,
    vmulq_f16_asm,
};
pub use attention_fused::{
    bf16_to_fp32_scalar, fp16_to_fp32_scalar, quantize_query_to_q8_0,
    quantize_query_to_q8_1, vec_dot_iq4_nl_q8_0, vec_dot_q4_0_q8_0,
    vec_dot_q4_1_q8_1, vec_dot_q5_0_q8_0, vec_dot_q5_1_q8_1, vec_dot_q8_0_q8_0,
    vec_mad_f32_iq4_nl, vec_mad_f32_q4_0, vec_mad_f32_q4_1, vec_mad_f32_q5_0,
    vec_mad_f32_q5_1, vec_mad_f32_q8_0,
};
pub use binary::{add_f32, div_f32, mul_f32, sub_f32};
pub use conv::{conv_2d_f32, pool_2d_avg_f32, pool_2x2, pool_3x3};
pub use dup::dup_f16_to_f32 as dup_f16_to_f32_neon;
pub use dup::dup_f32_to_f16 as dup_f32_to_f16_neon;
pub use gelu::gelu_f32 as gelu_f32_neon;
pub use gelu::gelu_quick_f32 as gelu_quick_f32_neon;
pub use moe::mul_mat_id_f32_neon;
pub use rms_norm::rms_norm_f32 as rms_norm_f32_neon;
pub use rms_norm::rms_norm_fp16_to_f32 as rms_norm_fp16_to_f32_neon;
pub use sampling::argmax_f32 as argmax_f32_neon;
pub use sampling::init_indices_neon;
pub use silu::silu_backward_f32 as silu_backward_f32_neon;
pub use silu::silu_f32 as silu_f32_neon;
pub use silu::vexpq_f32;
pub use softmax::exp as neon_exp;
pub use softmax::softmax_f32 as softmax_f32_neon;
pub use tensor::clamp_f32 as clamp_f32_neon;
pub use tensor::group_norm_f32 as group_norm_f32_neon;
pub use tensor::layer_norm_f32 as layer_norm_f32_neon;
