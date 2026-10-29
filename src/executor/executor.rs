// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use crate::buffer::{BufferId, BufferLocation, BufferManager, ComputeDevice};
use crate::executor::Op;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use crate::quant::vec_dot::gpu::MetalContext;

/// Operation Executor (Layer 2)
///
/// Executes compute operations across backends (CPU/GPU).
/// Automatically handles device selection and data movement.
pub struct OperationExecutor {
    /// Buffer manager for memory operations
    #[allow(dead_code)]
    buffer_mgr: std::sync::Arc<std::sync::Mutex<BufferManager>>,

    /// GPU context (Metal on macOS)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    gpu_ctx: std::sync::Arc<std::sync::Mutex<MetalContext>>,

    /// Preferred execution device
    preferred_device: ComputeDevice,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl OperationExecutor {
    pub fn new(
        buffer_mgr: std::sync::Arc<std::sync::Mutex<BufferManager>>,
        gpu_ctx: std::sync::Arc<std::sync::Mutex<MetalContext>>,
    ) -> Self {
        Self { buffer_mgr, gpu_ctx, preferred_device: ComputeDevice::LocalGpu }
    }

    /// Execute an operation
    ///
    /// Returns output buffer ID.
    /// Automatically handles device selection and fallback.
    pub fn execute(
        &mut self,
        op: &Op,
        inputs: &[BufferId],
        output: BufferId,
    ) -> BufferId {
        // Determine execution device
        let device = self.select_device(op);

        // Ensure inputs are on the selected device
        for input_id in inputs {
            self.ensure_buffer_on_device(*input_id, device);
        }

        // Execute on selected device
        match device {
            ComputeDevice::LocalGpu => self.execute_gpu(op, inputs, output),
            ComputeDevice::Cpu => self.execute_cpu(op, inputs, output),
        }
    }

    /// Select execution device based on operation and data
    fn select_device(&self, op: &Op) -> ComputeDevice {
        // GPU-first strategy
        if op.is_gpu_supported()
            && self.preferred_device == ComputeDevice::LocalGpu
        {
            ComputeDevice::LocalGpu
        } else {
            ComputeDevice::Cpu
        }
    }

    /// Ensure buffer is on specified device
    fn ensure_buffer_on_device(&mut self, id: BufferId, device: ComputeDevice) {
        let current = {
            let mgr = self.buffer_mgr.lock().unwrap();
            mgr.get_location(&id).cloned()
        };

        if let Some(loc) = current {
            let current_device = match loc {
                BufferLocation::Local(_) => ComputeDevice::LocalGpu,
                BufferLocation::Host(_) => ComputeDevice::Cpu,
                _ => return, // Remote/Pending - no action needed yet
            };

            if current_device != device {
                let mut mgr = self.buffer_mgr.lock().unwrap();
                mgr.transfer(id, current_device, device);
            }
        }
    }

    /// Execute operation on GPU
    fn execute_gpu(
        &mut self,
        op: &Op,
        inputs: &[BufferId],
        output: BufferId,
    ) -> BufferId {
        // For now, stub implementation
        // Actual GPU execution will be implemented with Layer 3 integration
        #[cfg(debug_assertions)]
        eprintln!("[OperationExecutor] GPU execution for {:?} (stub)", op);

        output
    }

    /// Execute operation on CPU
    fn execute_cpu(
        &mut self,
        op: &Op,
        inputs: &[BufferId],
        output: BufferId,
    ) -> BufferId {
        // For now, stub implementation
        // Actual CPU execution will be implemented with Layer 3 integration
        #[cfg(debug_assertions)]
        eprintln!("[OperationExecutor] CPU execution for {:?} (stub)", op);

        output
    }

    /// Set preferred execution device
    pub fn set_preferred_device(&mut self, device: ComputeDevice) {
        self.preferred_device = device;
    }

    /// Get preferred execution device
    pub fn preferred_device(&self) -> ComputeDevice {
        self.preferred_device
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
impl OperationExecutor {
    pub fn new(
        buffer_mgr: std::sync::Arc<std::sync::Mutex<BufferManager>>,
    ) -> Self {
        Self { buffer_mgr, preferred_device: ComputeDevice::Cpu }
    }

    pub fn execute(
        &mut self,
        op: &Op,
        inputs: &[BufferId],
        output: BufferId,
    ) -> BufferId {
        self.execute_cpu(op, inputs, output)
    }

    fn execute_cpu(
        &mut self,
        #[allow(unused_variables)] op: &Op,
        _inputs: &[BufferId],
        output: BufferId,
    ) -> BufferId {
        #[cfg(debug_assertions)]
        eprintln!("[OperationExecutor] CPU execution for {:?} (stub)", op);

        output
    }

    pub fn set_preferred_device(&mut self, device: ComputeDevice) {
        self.preferred_device = device;
    }

    pub fn preferred_device(&self) -> ComputeDevice {
        self.preferred_device
    }
}
