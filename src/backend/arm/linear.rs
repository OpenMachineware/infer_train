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
            unsafe {
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
        }
        QuantType::Q5_K => {
            let nb = (in_features + QK_K - 1) / QK_K;
            let mut input_q8k: Vec<BlockQ8_K> =
                vec![unsafe { std::mem::zeroed() }; nb];
            quant::quantize_q8_k(input, &mut input_q8k);
            unsafe {
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
        }
        QuantType::Q4_K => {
            let nb = (in_features + QK_K - 1) / QK_K;
            let mut input_q8k: Vec<BlockQ8_K> =
                vec![unsafe { std::mem::zeroed() }; nb];
            quant::quantize_q8_k(input, &mut input_q8k);
            unsafe {
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
        }
        QuantType::Q4_0 => {
            let nb = (in_features + QK4_0 - 1) / QK4_0;
            let mut input_q80: Vec<BlockQ8_0> =
                vec![BlockQ8_0 { d: 0, qs: [0; QK8_0] }; nb];
            quant::quantize_q8_0(input, &mut input_q80);
            unsafe {
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
                "Unsupported quantization type for linear_f32: {:?}",
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
        QuantType::Q4_0 => unsafe {
            gemv::gemv_q4_0_transposed(
                weight_data,
                weight_stride,
                input,
                output,
                out_features,
                in_features,
                n_threads,
            );
        },
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
        QuantType::Q6_K => unsafe {
            gemv::gemv_q6_k_transposed(
                weight_data,
                weight_stride,
                input,
                output,
                out_features,
                in_features,
                n_threads,
            );
        },
        QuantType::Q5_K => unsafe {
            gemv::gemv_q5_k_transposed(
                weight_data,
                weight_stride,
                input,
                output,
                out_features,
                in_features,
                n_threads,
            );
        },
        QuantType::Q4_K => unsafe {
            gemv::gemv_q4_k_transposed(
                weight_data,
                weight_stride,
                input,
                output,
                out_features,
                in_features,
                n_threads,
            );
        },
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

/// Batch linear layer with pre-quantized Q8_K input
pub fn linear_q8_k_batch(
    input: &[BlockQ8_K],
    weight_data: &[u8],
    weight_type: QuantType,
    weight_stride: usize,
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) -> Result<(), String> {
    match weight_type {
        QuantType::Q5_K => {
            gemv::mul_mat_q5_k(
                weight_data,
                weight_stride,
                input,
                output,
                out_features,
                in_features,
                n_tokens,
                n_threads,
            );
        }
        QuantType::Q6_K => {
            gemv::mul_mat_q6_k(
                weight_data,
                weight_stride,
                input,
                output,
                out_features,
                in_features,
                n_tokens,
                n_threads,
            );
        }
        QuantType::Q4_K => {
            gemv::mul_mat_q4_k(
                weight_data,
                weight_stride,
                input,
                output,
                out_features,
                in_features,
                n_tokens,
                n_threads,
            );
        }
        QuantType::IQ4_XS => {
            gemv::mul_mat_iq4_xs(
                weight_data,
                weight_stride,
                input,
                output,
                out_features,
                in_features,
                n_tokens,
                n_threads,
            );
        }
        _ => {
            return Err(format!(
                "Unsupported quantization type for linear_q8_k_batch: {:?}",
                weight_type
            ));
        }
    }

    Ok(())
}

/// Batch linear layer with pre-quantized Q8_0 input
pub fn linear_q8_0_batch(
    input: &[BlockQ8_0],
    weight_data: &[u8],
    weight_stride: usize,
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
) -> Result<(), String> {
    gemv::mul_mat_q4_0(
        weight_data,
        weight_stride,
        input,
        output,
        out_features,
        in_features,
        n_tokens,
        n_threads,
    );
    Ok(())
}

/// Batch linear layer with F32 input (quantizes internally)
pub fn linear_f32_batch(
    input: &[f32],
    weight_data: &[u8],
    weight_type: QuantType,
    weight_stride: usize,
    output: &mut [f32],
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    n_threads: usize,
    quant_buf_q8_0: &mut Vec<BlockQ8_0>,
    quant_buf_q8_k: &mut Vec<BlockQ8_K>,
) -> Result<(), String> {
    match weight_type {
        QuantType::F16 => {
            let stride_f16 = weight_stride / 2;
            for token_idx in 0..n_tokens {
                let in_start = token_idx * in_features;
                let out_start = token_idx * out_features;
                for i in 0..out_features {
                    let col_offset = i * stride_f16;
                    let mut sum = 0.0f32;
                    for j in 0..in_features {
                        let f16_val = half::f16::from_le_bytes([
                            weight_data[(col_offset + j) * 2],
                            weight_data[(col_offset + j) * 2 + 1],
                        ]);
                        sum += f16_val.to_f32() * input[in_start + j];
                    }
                    output[out_start + i] = sum;
                }
            }
        }
        QuantType::Q5_K | QuantType::Q6_K | QuantType::Q4_K => {
            let nb_per_token = (in_features + QK_K - 1) / QK_K;
            quant_buf_q8_k
                .resize(n_tokens * nb_per_token, unsafe { std::mem::zeroed() });

            // Parallel quantization using persistent thread pool
            // Each thread quantizes different tokens via work stealing
            if n_threads > 1 && n_tokens >= n_threads {
                #[repr(C)]
                struct QuantWork {
                    input_ptr: usize,
                    buf_ptr: usize,
                    in_feat: usize,
                    nb: usize,
                }

                let work = QuantWork {
                    input_ptr: input.as_ptr() as usize,
                    buf_ptr: quant_buf_q8_k.as_mut_ptr() as usize,
                    in_feat: in_features,
                    nb: nb_per_token,
                };

                unsafe fn quant_work_fn(
                    token_idx: usize,
                    _tid: usize,
                    data: *const u8,
                ) {
                    let work = &*(data as *const QuantWork);
                    let input_slice = std::slice::from_raw_parts(
                        (work.input_ptr as *const f32)
                            .add(token_idx * work.in_feat),
                        work.in_feat,
                    );
                    let buf_slice = std::slice::from_raw_parts_mut(
                        (work.buf_ptr as *mut crate::types::BlockQ8_K)
                            .add(token_idx * work.nb),
                        work.nb,
                    );
                    quant::quantize_q8_k(input_slice, buf_slice);
                }

                unsafe {
                    crate::backend::thread_pool::parallel_stealing_raw(
                        n_tokens,
                        n_threads,
                        quant_work_fn,
                        &work as *const _ as *const u8,
                    );
                }
            } else {
                // Sequential fallback
                for token_idx in 0..n_tokens {
                    let in_start = token_idx * in_features;
                    let out_start = token_idx * nb_per_token;
                    quant::quantize_q8_k(
                        &input[in_start..in_start + in_features],
                        &mut quant_buf_q8_k
                            [out_start..out_start + nb_per_token],
                    );
                }
            }

            match weight_type {
                QuantType::Q5_K => {
                    gemv::mul_mat_q5_k(
                        weight_data,
                        weight_stride,
                        quant_buf_q8_k,
                        output,
                        out_features,
                        in_features,
                        n_tokens,
                        n_threads,
                    );
                }
                QuantType::Q6_K => {
                    gemv::mul_mat_q6_k(
                        weight_data,
                        weight_stride,
                        quant_buf_q8_k,
                        output,
                        out_features,
                        in_features,
                        n_tokens,
                        n_threads,
                    );
                }
                QuantType::Q4_K => {
                    gemv::mul_mat_q4_k(
                        weight_data,
                        weight_stride,
                        quant_buf_q8_k,
                        output,
                        out_features,
                        in_features,
                        n_tokens,
                        n_threads,
                    );
                }
                _ => unreachable!(),
            }
        }
        QuantType::Q4_0 | QuantType::Q8_0 => {
            let nb_per_token = (in_features + QK8_0 - 1) / QK8_0;
            quant_buf_q8_0.resize(
                n_tokens * nb_per_token,
                BlockQ8_0 { d: 0, qs: [0; QK8_0] },
            );

            // Parallel quantization using persistent thread pool
            if n_threads > 1 && n_tokens >= n_threads {
                #[repr(C)]
                struct QuantWork {
                    input_ptr: usize,
                    buf_ptr: usize,
                    in_feat: usize,
                    nb: usize,
                }

                let work = QuantWork {
                    input_ptr: input.as_ptr() as usize,
                    buf_ptr: quant_buf_q8_0.as_mut_ptr() as usize,
                    in_feat: in_features,
                    nb: nb_per_token,
                };

                unsafe fn quant_work_fn(
                    token_idx: usize,
                    _tid: usize,
                    data: *const u8,
                ) {
                    let work = &*(data as *const QuantWork);
                    let input_slice = std::slice::from_raw_parts(
                        (work.input_ptr as *const f32)
                            .add(token_idx * work.in_feat),
                        work.in_feat,
                    );
                    let buf_slice = std::slice::from_raw_parts_mut(
                        (work.buf_ptr as *mut BlockQ8_0)
                            .add(token_idx * work.nb),
                        work.nb,
                    );
                    quant::quantize_q8_0(input_slice, buf_slice);
                }

                unsafe {
                    crate::backend::thread_pool::parallel_stealing_raw(
                        n_tokens,
                        n_threads,
                        quant_work_fn,
                        &work as *const _ as *const u8,
                    );
                }
            } else {
                // Sequential fallback
                for token_idx in 0..n_tokens {
                    let in_start = token_idx * in_features;
                    let out_start = token_idx * nb_per_token;
                    quant::quantize_q8_0(
                        &input[in_start..in_start + in_features],
                        &mut quant_buf_q8_0
                            [out_start..out_start + nb_per_token],
                    );
                }
            }

            gemv::mul_mat_q4_0(
                weight_data,
                weight_stride,
                quant_buf_q8_0,
                output,
                out_features,
                in_features,
                n_tokens,
                n_threads,
            );
        }
        _ => {
            return Err(format!(
                "Unsupported weight type for linear_f32_batch: {:?}",
                weight_type
            ));
        }
    }

    Ok(())
}
