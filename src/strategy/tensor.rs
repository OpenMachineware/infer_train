// Tensor parallel strategy

use crate::backend::BackendId;

/// Tensor parallel config
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
