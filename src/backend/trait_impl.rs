// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Backend trait definition.

use std::fmt;
use std::sync::Arc;

use super::device::{Device, DeviceId};
use super::memory::MemoryRegionId;
use super::operator::{CoreOperator, Cost, OperatorResult};

/// Data type for tensors.
/// Re-exported from model::tensor for unified type system.
pub use crate::model::tensor::DataType;

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
    fn supports_operator(&self, op: &CoreOperator) -> bool;

    /// Execute an operator.
    fn execute(
        &self,
        op: &CoreOperator,
        inputs: &[TensorRef],
        outputs: &mut [TensorRef],
    ) -> OperatorResult<()>;

    /// Estimate execution cost for an operator.
    fn estimate_cost(&self, op: &CoreOperator) -> Cost;
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
    /// Data pointer (F32 slice) - for activation tensors.
    pub data_f32: Option<*mut f32>,
    /// Data length (in elements) - for F32 data.
    pub len: usize,
    /// Raw byte data pointer - for quantized weight tensors.
    pub data_bytes: Option<*const u8>,
    /// Byte length - for raw data.
    pub byte_len: usize,
}

impl TensorRef {
    /// Create a new tensor reference.
    pub fn new(
        id: usize,
        shape: Vec<usize>,
        dtype: DataType,
        region: MemoryRegionId,
    ) -> Self {
        let len = shape.iter().product();
        Self {
            id,
            shape,
            dtype,
            region,
            data_f32: None,
            len,
            data_bytes: None,
            byte_len: 0,
        }
    }

    /// Create a tensor reference from F32 data.
    pub fn from_f32(id: usize, data: &mut [f32], shape: Vec<usize>) -> Self {
        let len = data.len();
        Self {
            id,
            shape,
            dtype: DataType::F32,
            region: MemoryRegionId(0),
            data_f32: Some(data.as_mut_ptr()),
            len,
            data_bytes: None,
            byte_len: 0,
        }
    }

    /// Create a tensor reference from quantized weight data.
    pub fn from_quantized(
        id: usize,
        data: &[u8],
        shape: Vec<usize>,
        dtype: DataType,
    ) -> Self {
        let len = shape.iter().product();
        Self {
            id,
            shape,
            dtype,
            region: MemoryRegionId(0),
            data_f32: None,
            len,
            data_bytes: Some(data.as_ptr()),
            byte_len: data.len(),
        }
    }

    /// Create from Tensor (for backward compatibility).
    pub fn from_tensor(id: usize, tensor: &crate::model::Tensor) -> Self {
        Self {
            id,
            shape: tensor.shape.clone(),
            dtype: tensor.dtype,
            region: MemoryRegionId(0),
            data_f32: None,
            len: tensor.shape.iter().product(),
            data_bytes: Some(tensor.bytes().as_ptr()),
            byte_len: tensor.bytes().len(),
        }
    }

    /// Get the number of elements.
    pub fn num_elements(&self) -> usize {
        self.len
    }

    /// Get the size in bytes.
    pub fn byte_size(&self) -> usize {
        if self.byte_len > 0 {
            self.byte_len
        } else {
            self.len * self.dtype.block_size()
        }
    }

    /// Get F32 data slice (mutable).
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        self.data_f32
            .map(|ptr| unsafe { std::slice::from_raw_parts_mut(ptr, self.len) })
    }

    /// Get F32 data slice (immutable).
    pub fn as_f32(&self) -> Option<&[f32]> {
        self.data_f32
            .map(|ptr| unsafe { std::slice::from_raw_parts(ptr, self.len) })
    }

    /// Get raw byte data.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        self.data_bytes.map(|ptr| unsafe {
            std::slice::from_raw_parts(ptr, self.byte_len)
        })
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
        op: &CoreOperator,
    ) -> Option<&Arc<dyn Backend>> {
        self.backends.iter().find(|b| b.supports_operator(op))
    }
}
