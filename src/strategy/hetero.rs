// 异构流水线策略

use crate::backend::BackendId;
use crate::strategy::ExecutionStrategy;

/// 异构流水线管理器
pub struct HeterogeneousPipeline {
    prefill_device: BackendId,
    decode_device: BackendId,
}

impl HeterogeneousPipeline {
    pub fn new(prefill: BackendId, decode: BackendId) -> Self {
        Self { prefill_device: prefill, decode_device: decode }
    }

    pub fn strategy(&self) -> ExecutionStrategy {
        ExecutionStrategy::HeterogeneousPipeline {
            prefill_device: self.prefill_device,
            decode_device: self.decode_device,
        }
    }
}
