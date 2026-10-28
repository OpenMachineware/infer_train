// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Single device execution strategy.

use std::sync::Arc;

use crate::backend::operator::{Cost, Operator, OperatorResult};
use crate::backend::{Backend, Device, DeviceId, MemoryRegionId};

/// Single device execution strategy.
///
/// Executes all operations on a single device (CPU or GPU).
#[derive(Debug)]
pub struct SingleDeviceStrategy {
    /// The backend to use.
    backend: Arc<dyn Backend>,
}

impl SingleDeviceStrategy {
    /// Create a new single device strategy.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        self.backend.device()
    }

    /// Get the device ID.
    pub fn device_id(&self) -> DeviceId {
        self.backend.device().id()
    }

    /// Get the memory region.
    pub fn memory_region(&self) -> MemoryRegionId {
        self.backend.memory_region()
    }

    /// Execute an operator.
    pub fn execute(
        &self,
        op: &dyn Operator,
        inputs: &[crate::backend::TensorRef],
        outputs: &mut [crate::backend::TensorRef],
    ) -> OperatorResult<()> {
        self.backend.execute(op, inputs, outputs)
    }

    /// Estimate execution cost.
    pub fn estimate_cost(&self, op: &dyn Operator) -> Cost {
        self.backend.estimate_cost(op)
    }
}
