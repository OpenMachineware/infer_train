// MoE 专家并行策略

use crate::backend::BackendId;
use std::collections::HashSet;

/// 专家放置策略
pub struct ExpertPlacement {
    /// 热门专家（常驻 GPU）
    hot_experts: HashSet<usize>,
    /// 热门专家设备
    hot_device: BackendId,
    /// 冷门专家设备
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
