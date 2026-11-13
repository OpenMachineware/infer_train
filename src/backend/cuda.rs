// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// NVIDIA CUDA backend

use crate::backend::Backend;
use crate::ops::{CoreOperator, OpError, QuantType};

pub struct CudaBackend {
    id: usize,
}

impl CudaBackend {
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

impl Backend for CudaBackend {
    fn id(&self) -> usize {
        self.id
    }
    fn name(&self) -> &str {
        "cuda"
    }

    fn available_memory(&self) -> usize {
        // TODO: detect actual VRAM
        24 * 1024 * 1024 * 1024 // 24GB
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
            CoreOperator::MoERouter { .. } => true,
            CoreOperator::MoEExpert { .. } => true,
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
        _inputs: &[&[f32]],
        _outputs: &mut [&mut [f32]],
    ) -> Result<(), OpError> {
        Err(OpError::NotImplemented(format!("{:?} on CUDA", op)))
    }
}
