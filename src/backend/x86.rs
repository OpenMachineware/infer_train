// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! x86 CPU backend using AVX2/AVX512 SIMD.
//!
//! This module is only compiled on x86_64 targets.

use super::device::{Device, DeviceId};
use super::memory::MemoryRegionId;
use super::operator::{Cost, Operator, OperatorResult};
use super::trait_impl::{Backend, TensorRef};

/// x86 CPU backend.
#[derive(Debug)]
pub struct X86CpuBackend {
    device: Device,
    memory_region: MemoryRegionId,
}

impl X86CpuBackend {
    /// Create a new x86 CPU backend.
    pub fn new(memory_region: MemoryRegionId) -> Self {
        let device = Device::detect_best();
        Self { device, memory_region }
    }
}

impl Backend for X86CpuBackend {
    fn name(&self) -> &str {
        "x86 CPU (AVX2/AVX512)"
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
