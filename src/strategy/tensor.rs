// 张量并行策略

use crate::backend::BackendId;

/// 张量并行配置
pub struct TensorParallel {
    devices: Vec<BackendId>,
}

impl TensorParallel {
    pub fn new(devices: Vec<BackendId>) -> Self {
        Self { devices }
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
}
