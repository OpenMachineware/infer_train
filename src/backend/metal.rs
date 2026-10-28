// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Metal GPU backend for Apple Silicon.
//!
//! This module is only compiled on aarch64 with metal feature.

use super::device::{Device, DeviceId};
use super::memory::MemoryRegionId;
use super::operator::{Cost, Operator, OperatorResult};
use super::trait_impl::{Backend, TensorRef};

/// Metal GPU backend.
#[derive(Debug)]
pub struct MetalBackend {
    device: Device,
    memory_region: MemoryRegionId,
}

impl MetalBackend {
    /// Create a new Metal backend.
    pub fn new(memory_region: MemoryRegionId) -> Self {
        let device = Device::detect_best();
        Self { device, memory_region }
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "Metal GPU"
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
        true
    }

    fn execute(
        &self,
        _op: &dyn Operator,
        _inputs: &[TensorRef],
        _outputs: &mut [TensorRef],
    ) -> OperatorResult<()> {
        Ok(())
    }

    fn estimate_cost(&self, _op: &dyn Operator) -> Cost {
        Cost::zero()
    }
}
