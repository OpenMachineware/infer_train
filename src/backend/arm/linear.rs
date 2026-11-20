// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Linear layer (GEMV) operations for CPU inference

use crate::backend::arm::{gemv, quant, vec_dot};
use crate::types::{BlockQ8_0, BlockQ8_K, QK_K, QK4_0, QK8_0, QuantType};

/// Linear layer with F32 input (quantizes internally)
pub fn linear_f32(
    input: &[f32],
    weight_data: &[u8],
    weight_type: QuantType,
    weight_stride: usize,
    output: &mut [f32],
    n_threads: usize,
) -> Result<(), String> {
    let in_features = input.len();
    let out_features = output.len();

    match weight_type {
        QuantType::F16 => {
            let stride_f16 = weight_stride / 2;
            for i in 0..out_features {
                let col_offset = i * stride_f16;
                let mut sum = 0.0f32;
                for j in 0..in_features {
                    let f16_val = half::f16::from_le_bytes([
                        weight_data[(col_offset + j) * 2],
                        weight_data[(col_offset + j) * 2 + 1],
                    ]);
                    sum += f16_val.to_f32() * input[j];
                }
                output[i] = sum;
            }
        }
        QuantType::Q6_K => {
            let nb = (in_features + QK_K - 1) / QK_K;
            let mut input_q8k: Vec<BlockQ8_K> =
                vec![unsafe { std::mem::zeroed() }; nb];
            quant::quantize_q8_k(input, &mut input_q8k);
            gemv::gemv_q6_k_transposed(
                weight_data,
                weight_stride,
                &input_q8k,
                output,
                out_features,
                in_features,
                n_threads,
            );
        }
        QuantType::Q5_K => {
            let nb = (in_features + QK_K - 1) / QK_K;
            let mut input_q8k: Vec<BlockQ8_K> =
                vec![unsafe { std::mem::zeroed() }; nb];
            quant::quantize_q8_k(input, &mut input_q8k);
            gemv::gemv_q5_k_transposed(
                weight_data,
                weight_stride,
                &input_q8k,
                output,
                out_features,
                in_features,
                n_threads,
            );
        }
        QuantType::Q4_K => {
            let nb = (in_features + QK_K - 1) / QK_K;
            let mut input_q8k: Vec<BlockQ8_K> =
                vec![unsafe { std::mem::zeroed() }; nb];
            quant::quantize_q8_k(input, &mut input_q8k);
            gemv::gemv_q4_k_transposed(
                weight_data,
                weight_stride,
                &input_q8k,
                output,
                out_features,
                in_features,
                n_threads,
            );
        }
        QuantType::Q4_0 => {
            let nb = (in_features + QK4_0 - 1) / QK4_0;
            let mut input_q80: Vec<BlockQ8_0> =
                vec![BlockQ8_0 { d: 0, qs: [0; QK8_0] }; nb];
            quant::quantize_q8_0(input, &mut input_q80);
            gemv::gemv_q4_0_transposed(
                weight_data,
                weight_stride,
                &input_q80,
                output,
                out_features,
                in_features,
                n_threads,
            );
        }
        QuantType::Q8_0 => {
            let nb = (in_features + QK8_0 - 1) / QK8_0;
            let mut input_q80: Vec<BlockQ8_0> =
                vec![BlockQ8_0 { d: 0, qs: [0; QK8_0] }; nb];
            quant::quantize_q8_0(input, &mut input_q80);
            for i in 0..out_features {
                let col_offset = i * weight_stride;
                let col_blocks: &[BlockQ8_0] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr().add(col_offset)
                            as *const BlockQ8_0,
                        nb,
                    )
                };
                output[i] = vec_dot::vec_dot_q8_0(col_blocks, &input_q80);
            }
        }
        _ => {
            return Err(format!(
                "Unsupported quantization type for linear: {:?}",
                weight_type
            ));
        }
    }

    Ok(())
}

/// Linear layer with pre-quantized Q8_0 input
pub fn linear_q8_0(
    input: &[BlockQ8_0],
    in_features: usize,
    weight_data: &[u8],
    weight_type: QuantType,
    weight_stride: usize,
    output: &mut [f32],
    n_threads: usize,
) -> Result<(), String> {
    let out_features = output.len();

    match weight_type {
        QuantType::Q4_0 => {
            gemv::gemv_q4_0_transposed(
                weight_data,
                weight_stride,
                input,
                output,
                out_features,
                in_features,
                n_threads,
            );
        }
        QuantType::Q8_0 => {
            let nb = (in_features + QK8_0 - 1) / QK8_0;
            for i in 0..out_features {
                let col_offset = i * weight_stride;
                let col_blocks: &[BlockQ8_0] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr().add(col_offset)
                            as *const BlockQ8_0,
                        nb,
                    )
                };
                output[i] = vec_dot::vec_dot_q8_0(col_blocks, input);
            }
        }
        _ => {
            return Err(format!(
                "Unsupported quantization type for linear_q8_0: {:?}",
                weight_type
            ));
        }
    }

    Ok(())
}

/// Linear layer with pre-quantized Q8_K input
pub fn linear_q8_k(
    input: &[BlockQ8_K],
    in_features: usize,
    weight_data: &[u8],
    weight_type: QuantType,
    weight_stride: usize,
    output: &mut [f32],
    n_threads: usize,
) -> Result<(), String> {
    let out_features = output.len();

    match weight_type {
        QuantType::Q6_K => {
            gemv::gemv_q6_k_transposed(
                weight_data,
                weight_stride,
                input,
                output,
                out_features,
                in_features,
                n_threads,
            );
        }
        QuantType::Q5_K => {
            gemv::gemv_q5_k_transposed(
                weight_data,
                weight_stride,
                input,
                output,
                out_features,
                in_features,
                n_threads,
            );
        }
        QuantType::Q4_K => {
            gemv::gemv_q4_k_transposed(
                weight_data,
                weight_stride,
                input,
                output,
                out_features,
                in_features,
                n_threads,
            );
        }
        QuantType::IQ4_XS => {
            let nb = (in_features + QK_K - 1) / QK_K;
            for i in 0..out_features {
                let col_offset = i * weight_stride;
                let col_blocks: &[crate::types::BlockIQ4XS] = unsafe {
                    std::slice::from_raw_parts(
                        weight_data.as_ptr().add(col_offset)
                            as *const crate::types::BlockIQ4XS,
                        nb,
                    )
                };
                output[i] = vec_dot::vec_dot_iq4_xs_q8_k(col_blocks, input);
            }
        }
        _ => {
            return Err(format!(
                "Unsupported quantization type for linear_q8_k: {:?}",
                weight_type
            ));
        }
    }

    Ok(())
}
