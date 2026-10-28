// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Backend trait definition.

use std::fmt;
use std::sync::Arc;

use super::device::{Device, DeviceId};
use super::memory::MemoryRegionId;
use super::operator::{Cost, Operator, OperatorResult};

/// Backend trait for compute operations.
///
/// This trait provides a hardware-agnostic interface for tensor operations,
/// supporting CPU and GPU backends with various architectures.
pub trait Backend: Send + Sync + fmt::Debug {
    /// Get the backend name.
    fn name(&self) -> &str;

    /// Get the device this backend runs on.
    fn device(&self) -> &Device;

    /// Get the memory region used by this backend.
    fn memory_region(&self) -> MemoryRegionId;

    /// Get available memory in bytes.
    fn available_memory(&self) -> u64;

    /// Check if this backend supports a specific operator.
    fn supports_operator(&self, op: &dyn Operator) -> bool;

    /// Execute an operator.
    fn execute(
        &self,
        op: &dyn Operator,
        inputs: &[TensorRef],
        outputs: &mut [TensorRef],
    ) -> OperatorResult<()>;

    /// Estimate execution cost for an operator.
    fn estimate_cost(&self, op: &dyn Operator) -> Cost;
}

/// Reference to a tensor.
#[derive(Clone, Debug)]
pub struct TensorRef {
    /// Tensor ID.
    pub id: usize,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Tensor data type.
    pub dtype: DataType,
    /// Memory region.
    pub region: MemoryRegionId,
}

impl TensorRef {
    /// Create a new tensor reference.
    pub fn new(
        id: usize,
        shape: Vec<usize>,
        dtype: DataType,
        region: MemoryRegionId,
    ) -> Self {
        Self { id, shape, dtype, region }
    }

    /// Get the number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the size in bytes.
    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.dtype.size()
    }
}

/// Data type for tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    F32,
    F16,
    BF16,
    I32,
    I16,
    I8,
    U8,
}

impl DataType {
    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 => 2,
            DataType::BF16 => 2,
            DataType::I32 => 4,
            DataType::I16 => 2,
            DataType::I8 => 1,
            DataType::U8 => 1,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::F32 => write!(f, "f32"),
            DataType::F16 => write!(f, "f16"),
            DataType::BF16 => write!(f, "bf16"),
            DataType::I32 => write!(f, "i32"),
            DataType::I16 => write!(f, "i16"),
            DataType::I8 => write!(f, "i8"),
            DataType::U8 => write!(f, "u8"),
        }
    }
}

/// Backend registry for managing multiple backends.
#[derive(Debug, Default)]
pub struct BackendRegistry {
    backends: Vec<Arc<dyn Backend>>,
}

impl BackendRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self { backends: Vec::new() }
    }

    /// Register a backend.
    pub fn register(&mut self, backend: Arc<dyn Backend>) {
        self.backends.push(backend);
    }

    /// Get all backends.
    pub fn all(&self) -> &[Arc<dyn Backend>] {
        &self.backends
    }

    /// Get a backend by device ID.
    pub fn get_by_device(
        &self,
        device_id: DeviceId,
    ) -> Option<&Arc<dyn Backend>> {
        self.backends.iter().find(|b| b.device().id() == device_id)
    }

    /// Get the best backend for an operator.
    pub fn best_for_operator(
        &self,
        op: &dyn Operator,
    ) -> Option<&Arc<dyn Backend>> {
        self.backends.iter().find(|b| b.supports_operator(op))
    }
}
