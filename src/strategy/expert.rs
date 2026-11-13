// SPDX-License-Identifier: MIT

// MoE expert parallel strategy

use crate::backend::BackendId;
use std::collections::HashSet;

/// Expert placement strategy
pub struct ExpertPlacement {
    /// Hot experts (resident on GPU)
    hot_experts: HashSet<usize>,
    /// Hot expert device
    hot_device: BackendId,
    /// Cold expert device
    cold_device: BackendId,
}

impl ExpertPlacement {
    pub fn new(hot_device: BackendId, cold_device: BackendId) -> Self {
        Self { hot_experts: HashSet::new(), hot_device, cold_device }
    }

    pub fn is_hot(&self, expert_id: usize) -> bool {
        self.hot_experts.contains(&expert_id)
    }

    pub fn mark_hot(&mut self, expert_id: usize) {
        self.hot_experts.insert(expert_id);
    }

    pub fn hot_device(&self) -> BackendId {
        self.hot_device
    }

    pub fn cold_device(&self) -> BackendId {
        self.cold_device
    }
}
