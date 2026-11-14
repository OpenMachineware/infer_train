// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// ARM NEON backend

pub mod activation;
pub mod flash_attn;
pub mod gemv;
pub mod rms_norm;
pub mod rope;
pub mod softmax;
pub mod vec_dot;

use crate::backend::{Backend, TensorRef, TensorRefMut};
use crate::types::*;

pub struct ArmCpuBackend {
    id: usize,
}

impl ArmCpuBackend {
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

impl Backend for ArmCpuBackend {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &str {
        "arm"
    }

    fn available_memory(&self) -> usize {
        16 * 1024 * 1024 * 1024 // 16GB
    }

    fn supports_op(&self, op: &CoreOperator) -> bool {
        match op {
            CoreOperator::RmsNorm { .. } => true,
            CoreOperator::Linear { quant_type, .. } => {
                matches!(
                    quant_type,
                    QuantType::Q4_0
                        | QuantType::Q4_1
                        | QuantType::Q5_0
                        | QuantType::Q5_1
                        | QuantType::Q8_0
                        | QuantType::Q8_1
                        | QuantType::Q4_K
                        | QuantType::Q5_K
                        | QuantType::Q6_K
                        | QuantType::Q8_K
                )
            }
            CoreOperator::Rope { .. } => true,
            CoreOperator::FlashAttention { .. } => true,
            CoreOperator::SwiGLU { .. } => true,
            CoreOperator::MoERouter { .. } => false,
            CoreOperator::MoEExpert { .. } => false,
        }
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::Q4_0
                | QuantType::Q4_1
                | QuantType::Q5_0
                | QuantType::Q5_1
                | QuantType::Q8_0
                | QuantType::Q8_1
                | QuantType::Q4_K
                | QuantType::Q5_K
                | QuantType::Q6_K
                | QuantType::Q8_K
        )
    }

    fn execute(
        &self,
        op: &CoreOperator,
        inputs: &[TensorRef],
        outputs: &mut [TensorRefMut],
    ) -> Result<(), OpError> {
        match op {
            CoreOperator::RmsNorm { hidden_size, eps } => {
                if inputs.len() != 2 || outputs.len() != 1 {
                    return Err(OpError::InvalidInput(
                        "RmsNorm requires 2 inputs and 1 output".into(),
                    ));
                }

                let x = unsafe {
                    std::slice::from_raw_parts(
                        inputs[0].data as *const f32,
                        *hidden_size,
                    )
                };
                let weight = unsafe {
                    std::slice::from_raw_parts(
                        inputs[1].data as *const f32,
                        *hidden_size,
                    )
                };
                let output = unsafe {
                    std::slice::from_raw_parts_mut(
                        outputs[0].data as *mut f32,
                        *hidden_size,
                    )
                };

                rms_norm::rms_norm_f32(x, weight, output, *eps);
                Ok(())
            }

            CoreOperator::Linear { in_features, out_features, quant_type } => {
                if inputs.len() != 2 || outputs.len() != 1 {
                    return Err(OpError::InvalidInput(
                        "Linear requires 2 inputs and 1 output".into(),
                    ));
                }

                let output = unsafe {
                    std::slice::from_raw_parts_mut(
                        outputs[0].data as *mut f32,
                        *out_features,
                    )
                };

                match quant_type {
                    QuantType::Q4_0 => {
                        let blocks_per_row = (*in_features + QK8_0 - 1) / QK8_0;
                        let weights = unsafe {
                            std::slice::from_raw_parts(
                                inputs[0].data as *const BlockQ4_0,
                                *out_features * blocks_per_row,
                            )
                        };

                        use crate::backend::TensorType;
                        let input_blocks =
                            if inputs[1].dtype == TensorType::Q8_0 {
                                (*in_features + QK8_0 - 1) / QK8_0
                            } else {
                                return Err(OpError::InvalidInput(
                                    "Linear Q4_0 requires Q8_0 input".into(),
                                ));
                            };
                        let input = unsafe {
                            std::slice::from_raw_parts(
                                inputs[1].data as *const BlockQ8_0,
                                input_blocks,
                            )
                        };

                        gemv::gemv_q4_0(
                            weights,
                            input,
                            output,
                            *out_features,
                            *in_features,
                        );
                        Ok(())
                    }

                    QuantType::Q8_0 => {
                        let blocks_per_row = (*in_features + QK8_0 - 1) / QK8_0;
                        let weights = unsafe {
                            std::slice::from_raw_parts(
                                inputs[0].data as *const BlockQ8_0,
                                *out_features * blocks_per_row,
                            )
                        };

                        let input = unsafe {
                            std::slice::from_raw_parts(
                                inputs[1].data as *const BlockQ8_0,
                                blocks_per_row,
                            )
                        };

                        gemv::gemv_q8_0(
                            weights,
                            input,
                            output,
                            *out_features,
                            *in_features,
                        );
                        Ok(())
                    }

                    QuantType::Q4_K => {
                        let blocks_per_row = (*in_features + QK_K - 1) / QK_K;
                        let weights = unsafe {
                            std::slice::from_raw_parts(
                                inputs[0].data as *const BlockQ4_K,
                                *out_features * blocks_per_row,
                            )
                        };

                        let input = unsafe {
                            std::slice::from_raw_parts(
                                inputs[1].data as *const BlockQ8_K,
                                blocks_per_row,
                            )
                        };

                        gemv::gemv_q4_k(
                            weights,
                            input,
                            output,
                            *out_features,
                            *in_features,
                        );
                        Ok(())
                    }

                    QuantType::Q5_K => {
                        let blocks_per_row = (*in_features + QK_K - 1) / QK_K;
                        let weights = unsafe {
                            std::slice::from_raw_parts(
                                inputs[0].data as *const BlockQ5_K,
                                *out_features * blocks_per_row,
                            )
                        };

                        let input = unsafe {
                            std::slice::from_raw_parts(
                                inputs[1].data as *const BlockQ8_K,
                                blocks_per_row,
                            )
                        };

                        gemv::gemv_q5_k(
                            weights,
                            input,
                            output,
                            *out_features,
                            *in_features,
                        );
                        Ok(())
                    }

                    QuantType::Q6_K => {
                        let blocks_per_row = (*in_features + QK_K - 1) / QK_K;
                        let weights = unsafe {
                            std::slice::from_raw_parts(
                                inputs[0].data as *const BlockQ6_K,
                                *out_features * blocks_per_row,
                            )
                        };

                        let input = unsafe {
                            std::slice::from_raw_parts(
                                inputs[1].data as *const BlockQ8_K,
                                blocks_per_row,
                            )
                        };

                        gemv::gemv_q6_k(
                            weights,
                            input,
                            output,
                            *out_features,
                            *in_features,
                        );
                        Ok(())
                    }

                    _ => Err(OpError::NotImplemented(format!(
                        "Linear with quant type {:?}",
                        quant_type
                    ))),
                }
            }

            CoreOperator::Rope { head_dim, max_seq_len: _ } => {
                if inputs.len() != 1 || outputs.len() != 1 {
                    return Err(OpError::InvalidInput(
                        "Rope requires 1 input and 1 output".into(),
                    ));
                }

                let n = inputs[0].dims[0];
                let x = unsafe {
                    std::slice::from_raw_parts(inputs[0].data as *const f32, n)
                };
                let output = unsafe {
                    std::slice::from_raw_parts_mut(
                        outputs[0].data as *mut f32,
                        n,
                    )
                };

                rope::rope_f32(x, output, 0, *head_dim, *head_dim, 10000.0);
                Ok(())
            }

            CoreOperator::FlashAttention {
                n_heads,
                n_kv_heads: _,
                head_dim,
                max_seq_len: _,
            } => {
                if inputs.len() < 3 || outputs.len() != 1 {
                    return Err(OpError::InvalidInput(
                        "FlashAttention requires 3+ inputs and 1 output".into(),
                    ));
                }

                let q = unsafe {
                    std::slice::from_raw_parts(
                        inputs[0].data as *const f32,
                        n_heads * head_dim,
                    )
                };
                let n_kv = inputs[1].dims[1];
                let k = unsafe {
                    std::slice::from_raw_parts(
                        inputs[1].data as *const f32,
                        n_kv * head_dim,
                    )
                };
                let v = unsafe {
                    std::slice::from_raw_parts(
                        inputs[2].data as *const f32,
                        n_kv * head_dim,
                    )
                };
                let output = unsafe {
                    std::slice::from_raw_parts_mut(
                        outputs[0].data as *mut f32,
                        n_heads * head_dim,
                    )
                };

                for h in 0..*n_heads {
                    let q_head = &q[h * head_dim..(h + 1) * head_dim];
                    let out_head =
                        &mut output[h * head_dim..(h + 1) * head_dim];

                    let mask = if inputs.len() > 3 {
                        Some(unsafe {
                            std::slice::from_raw_parts(
                                inputs[3].data as *const f32,
                                n_kv,
                            )
                        })
                    } else {
                        None
                    };

                    flash_attn::flash_attn_f32(
                        q_head,
                        k,
                        v,
                        mask,
                        out_head,
                        *head_dim,
                        n_kv,
                        1.0 / (*head_dim as f32).sqrt(),
                    );
                }
                Ok(())
            }

            CoreOperator::SwiGLU { hidden_size, intermediate_size } => {
                if inputs.len() != 2 || outputs.len() != 1 {
                    return Err(OpError::InvalidInput(
                        "SwiGLU requires 2 inputs and 1 output".into(),
                    ));
                }

                let gate = unsafe {
                    std::slice::from_raw_parts(
                        inputs[0].data as *const f32,
                        *intermediate_size,
                    )
                };
                let up = unsafe {
                    std::slice::from_raw_parts(
                        inputs[1].data as *const f32,
                        *intermediate_size,
                    )
                };

                let mut gate_silu = vec![0.0f32; *intermediate_size];
                activation::silu_f32(gate, &mut gate_silu);

                let intermediate: Vec<f32> = gate_silu
                    .iter()
                    .zip(up.iter())
                    .map(|(g, u)| g * u)
                    .collect();

                let output = unsafe {
                    std::slice::from_raw_parts_mut(
                        outputs[0].data as *mut f32,
                        *hidden_size,
                    )
                };

                output.copy_from_slice(&intermediate[..*hidden_size]);

                Ok(())
            }

            _ => Err(OpError::NotImplemented(format!("{:?}", op))),
        }
    }
}
