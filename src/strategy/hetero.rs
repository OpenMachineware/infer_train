// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Heterogeneous pipeline strategy

use crate::backend::BackendId;
use crate::strategy::ExecutionStrategy;

/// Heterogeneous pipeline manager
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
