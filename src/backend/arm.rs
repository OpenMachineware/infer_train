// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// ARM NEON backend

use crate::backend::Backend;
use crate::ops::{CoreOperator, OpError, QuantType};

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
        // TODO: actual detection
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
        match op {
            CoreOperator::RmsNorm { .. } => {
                Err(OpError::NotImplemented("RmsNorm on ARM".into()))
            }
            CoreOperator::Linear { .. } => {
                Err(OpError::NotImplemented("Linear on ARM".into()))
            }
            _ => Err(OpError::NotImplemented(format!("{:?}", op))),
        }
    }
}
