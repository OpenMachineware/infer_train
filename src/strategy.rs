// 执行策略

pub mod expert;
pub mod hetero;
pub mod tensor;

use crate::backend::BackendId;

/// 执行策略
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    /// 单设备执行
    SingleDevice { device: BackendId },

    /// 异构流水线（CPU prefill + GPU decode）
    HeterogeneousPipeline {
        prefill_device: BackendId,
        decode_device: BackendId,
    },

    /// 张量并行（多 GPU 同构）
    TensorParallel { devices: Vec<BackendId> },

    /// 专家并行（MoE 异构）
    ExpertParallel {
        hot_expert_device: BackendId,
        cold_expert_device: BackendId,
    },
}
