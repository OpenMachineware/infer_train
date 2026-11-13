// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
//
// Linear operator with quantization support

use crate::backend::operator::{Operator, OperatorId, OperatorResult};
use crate::cpu::CpuWorkBuffers;
use crate::model::tensor::DataType;
use std::fmt;

/// Linear (GEMV) operator for quantized weights
///
/// Supports:
/// - K-quant weights (Q4_K, Q5_K, Q6_K)
/// - Pre-quantized input (shared across multiple projections)
/// - Hybrid single-thread/multi-thread execution
pub struct LinearOp {
    id: OperatorId,
    in_features: usize,
    out_features: usize,
    weight_dtype: DataType,
}

impl LinearOp {
    /// Create a new linear operator
    pub fn new(
        id: OperatorId,
        in_features: usize,
        out_features: usize,
        weight_dtype: DataType,
    ) -> Self {
        Self { id, in_features, out_features, weight_dtype }
    }

    /// Execute the linear operator with pre-quantized input
    ///
    /// This is the key optimization:
    /// - Input is quantized once and reused for multiple projections
    /// - Uses hybrid ST/MT GEMV (m < 2048: single-thread, m >= 2048: multi-thread)
    #[cfg(target_arch = "aarch64")]
    pub fn execute_with_prequant(
        &self,
        x_q8k: &[crate::quant::types::BlockQ8K],
        weights_bytes: &[u8],
        output: &mut [f32],
        _buffers: &mut CpuWorkBuffers,
    ) -> OperatorResult<()> {
        use crate::quant::matmul::k_quant::{
            matmul_q4_k_q8_k_hybrid, matmul_q5_k_q8_k_hybrid,
            matmul_q6_k_q8_k_hybrid,
        };
        use crate::quant::types::{BlockQ4K, BlockQ5K, BlockQ6K, QK_K};

        let in_size_padded = ((self.in_features + QK_K - 1) / QK_K) * QK_K;
        let nb = in_size_padded / QK_K;

        // Verify input size
        if x_q8k.len() != nb {
            return Err(
                crate::backend::operator::OperatorError::InvalidInputShape {
                    expected: format!("{} blocks", nb),
                    actual: format!("{} blocks", x_q8k.len()),
                },
            );
        }

        // Verify output size
        if output.len() != self.out_features {
            return Err(
                crate::backend::operator::OperatorError::InvalidOutputShape {
                    expected: format!("{}", self.out_features),
                    actual: format!("{}", output.len()),
                },
            );
        }

        // Dispatch to appropriate kernel based on weight dtype
        unsafe {
            match self.weight_dtype {
                DataType::Q4_K => {
                    let weights_slice = std::slice::from_raw_parts(
                        weights_bytes.as_ptr() as *const BlockQ4K,
                        self.out_features * nb,
                    );
                    matmul_q4_k_q8_k_hybrid(
                        weights_slice,
                        x_q8k,
                        output,
                        in_size_padded,
                        self.out_features,
                    );
                }
                DataType::Q5_K => {
                    let weights_slice = std::slice::from_raw_parts(
                        weights_bytes.as_ptr() as *const BlockQ5K,
                        self.out_features * nb,
                    );
                    matmul_q5_k_q8_k_hybrid(
                        weights_slice,
                        x_q8k,
                        output,
                        in_size_padded,
                        self.out_features,
                    );
                }
                DataType::Q6_K => {
                    let weights_slice = std::slice::from_raw_parts(
                        weights_bytes.as_ptr() as *const BlockQ6K,
                        self.out_features * nb,
                    );
                    matmul_q6_k_q8_k_hybrid(
                        weights_slice,
                        x_q8k,
                        output,
                        in_size_padded,
                        self.out_features,
                    );
                }
                _ => {
                    return Err(crate::backend::operator::OperatorError::UnsupportedOperation(
                        format!("Unsupported weight dtype: {:?}", self.weight_dtype)
                    ));
                }
            }
        }

        Ok(())
    }

    /// Quantize input to Q8_K format
    ///
    /// This can be shared across multiple linear operators with the same input
    #[cfg(target_arch = "aarch64")]
    pub fn quantize_input<'a>(
        input: &[f32],
        buffers: &'a mut CpuWorkBuffers,
    ) -> &'a [crate::quant::types::BlockQ8K] {
        use crate::model::llama::LlamaModel;
        use crate::quant::types::{BlockQ8K, QK_K};

        let in_size = input.len();
        let in_size_padded = ((in_size + QK_K - 1) / QK_K) * QK_K;
        let nb = in_size_padded / QK_K;

        // Get quantization buffer
        let x_padded = &mut buffers.x_padded[..in_size_padded];
        let q8k_blocks = unsafe {
            std::slice::from_raw_parts_mut(
                buffers.q8k_input.as_mut_ptr() as *mut BlockQ8K,
                nb,
            )
        };

        // Quantize
        LlamaModel::quantize_input_to_q8k(input, q8k_blocks, x_padded);

        q8k_blocks
    }
}

impl Operator for LinearOp {
    fn id(&self) -> OperatorId {
        self.id
    }

    fn name(&self) -> &str {
        "Linear"
    }

    fn input_shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.in_features]]
    }

    fn output_shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.out_features]]
    }
}

impl fmt::Debug for LinearOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinearOp")
            .field("id", &self.id)
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("weight_dtype", &self.weight_dtype)
            .finish()
    }
}
