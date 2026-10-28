// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! ARM CPU backend using NEON SIMD.

use super::device::Device;
use super::memory::MemoryRegionId;
use super::operator::{Cost, Operator, OperatorResult};
use super::trait_impl::{Backend, TensorRef};

/// ARM CPU backend.
#[derive(Debug)]
pub struct ArmCpuBackend {
    device: Device,
    memory_region: MemoryRegionId,
}

impl ArmCpuBackend {
    /// Create a new ARM CPU backend.
    pub fn new(memory_region: MemoryRegionId) -> Self {
        let device = Device::detect_best();
        Self { device, memory_region }
    }
}

impl Backend for ArmCpuBackend {
    fn name(&self) -> &str {
        "ARM CPU (NEON)"
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn memory_region(&self) -> MemoryRegionId {
        self.memory_region
    }

    fn available_memory(&self) -> u64 {
        self.device.memory_size()
    }

    fn supports_operator(&self, _op: &dyn Operator) -> bool {
        // TODO: Check operator support
        true
    }

    fn execute(
        &self,
        _op: &dyn Operator,
        _inputs: &[TensorRef],
        _outputs: &mut [TensorRef],
    ) -> OperatorResult<()> {
        // TODO: Implement operator execution
        Ok(())
    }

    fn estimate_cost(&self, _op: &dyn Operator) -> Cost {
        // TODO: Implement cost estimation
        Cost::zero()
    }
}
