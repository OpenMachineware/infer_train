// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Execution strategies for heterogeneous systems.

use std::sync::Arc;

use crate::backend::operator::{CoreOperator, OperatorResult};
use crate::backend::{
    Backend, DeviceId, MemoryRegion, MemoryTopology, TensorRef,
};

/// Transfer strategy for KV cache between devices.
#[derive(Clone, Debug)]
pub enum TransferStrategy {
    /// No transfer needed (same device).
    None,
    /// Copy on demand.
    OnDemand,
    /// Pre-copy to destination.
    PreCopy,
    /// Stream asynchronously.
    Async,
}

/// Execution strategy for inference.
#[derive(Debug)]
pub enum ExecutionStrategy {
    /// Single device execution (CPU or GPU).
    SingleDevice {
        /// Device to use.
        device: DeviceId,
        /// Backend.
        backend: Arc<dyn Backend>,
    },

    /// Heterogeneous pipeline (CPU-GPU async).
    HeterogeneousPipeline {
        /// Prefill device (usually GPU for throughput).
        prefill_device: DeviceId,
        /// Decode device (CPU for latency, GPU for throughput).
        decode_device: DeviceId,
        /// KV cache transfer strategy.
        kv_transfer: TransferStrategy,
        /// Prefill backend.
        prefill_backend: Arc<dyn Backend>,
        /// Decode backend.
        decode_backend: Arc<dyn Backend>,
    },

    /// Tensor parallel (multi-GPU).
    TensorParallel {
        /// Devices to use.
        devices: Vec<DeviceId>,
        /// Backends for each device.
        backends: Vec<Arc<dyn Backend>>,
    },

    /// Expert parallel (MoE heterogeneous).
    ExpertParallel {
        /// Hot expert device (fast memory).
        hot_expert_device: DeviceId,
        /// Cold expert device (large memory).
        cold_expert_device: DeviceId,
    },

    /// Pipeline parallel (multi-node).
    PipelineParallel {
        /// Pipeline stages.
        stages: Vec<PipelineStage>,
    },
}

/// Pipeline stage for multi-node inference.
#[derive(Clone, Debug)]
pub struct PipelineStage {
    /// Layer range for this stage.
    pub layer_start: usize,
    pub layer_end: usize,
    /// Device for this stage.
    pub device: DeviceId,
}

impl ExecutionStrategy {
    /// Derive the best execution strategy from memory topology.
    pub fn derive(topology: &MemoryTopology) -> Self {
        // Check available regions
        let unified_count =
            topology.regions.iter().filter(|r| r.is_unified()).count();

        let gpu_count =
            topology.regions.iter().filter(|r| r.gpu_accessible()).count();

        if unified_count > 0 && gpu_count > 0 {
            // Apple Silicon or AI Max: unified memory
            // Prefer GPU for both prefill and decode
            // Use CPU for some operations if needed
            Self::select_unified_strategy(topology)
        } else if gpu_count > 1 {
            // Multiple GPUs: tensor parallel
            Self::select_tensor_parallel_strategy(topology)
        } else {
            // Single device
            Self::select_single_device_strategy(topology)
        }
    }

    fn select_unified_strategy(topology: &MemoryTopology) -> Self {
        // For unified memory, we could use heterogeneous pipeline
        // But for now, use single GPU device
        Self::select_single_device_strategy(topology)
    }

    fn select_tensor_parallel_strategy(_topology: &MemoryTopology) -> Self {
        // TODO: Implement tensor parallel strategy
        // For now, fall back to single device
        Self::select_single_device_strategy(_topology)
    }

    fn select_single_device_strategy(topology: &MemoryTopology) -> Self {
        // Find the first GPU region, or fall back to CPU
        let gpu_region = topology.regions.iter().find(|r| r.gpu_accessible());

        let cpu_region = topology.regions.iter().find(|r| r.cpu_accessible());

        let region = gpu_region.or(cpu_region).unwrap_or(&topology.regions[0]);

        let memory_region_id = region.id();

        // Get device ID from region if available
        let device_id = match region {
            MemoryRegion::Unified { device_id, .. } => *device_id,
            MemoryRegion::DeviceLocal { device_id, .. } => *device_id,
            _ => crate::backend::DeviceId(0),
        };

        // Create appropriate backend
        #[cfg(all(target_arch = "aarch64", feature = "metal"))]
        let backend: Arc<dyn Backend> =
            Arc::new(crate::backend::MetalBackend::new(memory_region_id));

        #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
        let backend: Arc<dyn Backend> =
            Arc::new(crate::backend::ArmCpuBackend::new(memory_region_id));

        Self::SingleDevice { device: device_id, backend }
    }

    /// Execute an operator.
    pub fn execute(
        &self,
        op: &CoreOperator,
        inputs: &[TensorRef],
        outputs: &mut [TensorRef],
    ) -> OperatorResult<()> {
        match self {
            ExecutionStrategy::SingleDevice { backend, .. } => {
                backend.execute(op, inputs, outputs)
            }
            ExecutionStrategy::HeterogeneousPipeline {
                prefill_backend,
                decode_backend: _,
                ..
            } => {
                // For now, just use prefill backend
                // Real implementation would check if this is prefill or decode
                prefill_backend.execute(op, inputs, outputs)
            }
            _ => Err(
                crate::backend::operator::OperatorError::UnsupportedOperation(
                    "Strategy not yet implemented".into(),
                ),
            ),
        }
    }

    /// Get the primary backend.
    pub fn primary_backend(&self) -> Option<&Arc<dyn Backend>> {
        match self {
            ExecutionStrategy::SingleDevice { backend, .. } => Some(backend),
            ExecutionStrategy::HeterogeneousPipeline {
                prefill_backend,
                ..
            } => Some(prefill_backend),
            _ => None,
        }
    }
}
