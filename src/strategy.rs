// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Execution strategy

pub mod expert;
pub mod hetero;
pub mod tensor;

use crate::backend::BackendId;

/// Execution strategy
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    /// Single device execution
    SingleDevice { device: BackendId },

    /// Heterogeneous pipeline (CPU prefill + GPU decode)
    HeterogeneousPipeline {
        prefill_device: BackendId,
        decode_device: BackendId,
    },

    /// Tensor parallel (multi-GPU homogeneous)
    TensorParallel { devices: Vec<BackendId> },

    /// Expert parallel (MoE heterogeneous)
    ExpertParallel {
        hot_expert_device: BackendId,
        cold_expert_device: BackendId,
    },
}
