// Apple Metal 后端

use crate::backend::Backend;
use crate::ops::{CoreOperator, OpError, QuantType};

pub struct MetalBackend {
    id: usize,
}

impl MetalBackend {
    pub fn new(id: usize) -> Self {
        Self { id }
    }
}

impl Backend for MetalBackend {
    fn id(&self) -> usize {
        self.id
    }
    fn name(&self) -> &str {
        "metal"
    }

    fn available_memory(&self) -> usize {
        // TODO: 实际检测统一内存大小
        128 * 1024 * 1024 * 1024 // 128GB (统一内存)
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
        Err(OpError::NotImplemented(format!("{:?} on Metal", op)))
    }
}
