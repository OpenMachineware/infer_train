// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Backend trait for executing compute graphs

use super::device::Device;
use super::graph::ComputeGraph;
use super::tensor::Tensor;
use crate::model::tensor::DataType;

/// Result type for backend operations
pub type BackendResult<T> = Result<T, BackendError>;

/// Errors that can occur during backend operations
#[derive(Debug, Clone)]
pub enum BackendError {
    /// Tensor not found
    TensorNotFound(String),
    /// Operation not supported
    OpNotSupported(String),
    /// Device mismatch
    DeviceMismatch { expected: Device, actual: Device },
    /// Shape mismatch
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    /// Dtype mismatch
    DtypeMismatch { expected: DataType, actual: DataType },
    /// Memory allocation failed
    AllocFailed(String),
    /// Execution failed
    ExecFailed(String),
    /// Invalid operation parameters
    InvalidParams(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::TensorNotFound(name) => {
                write!(f, "Tensor not found: {}", name)
            }
            BackendError::OpNotSupported(op) => {
                write!(f, "Operation not supported: {}", op)
            }
            BackendError::DeviceMismatch { expected, actual } => {
                write!(
                    f,
                    "Device mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            BackendError::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Shape mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            BackendError::DtypeMismatch { expected, actual } => {
                write!(
                    f,
                    "Dtype mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            BackendError::AllocFailed(msg) => {
                write!(f, "Allocation failed: {}", msg)
            }
            BackendError::ExecFailed(msg) => {
                write!(f, "Execution failed: {}", msg)
            }
            BackendError::InvalidParams(msg) => {
                write!(f, "Invalid parameters: {}", msg)
            }
        }
    }
}

impl std::error::Error for BackendError {}

/// Backend trait for executing operations on a device
/// Note: Not requiring Send+Sync to allow Rc-based implementations (Metal)
pub trait Backend: std::fmt::Debug {
    /// Get the device this backend runs on
    fn device(&self) -> Device;

    /// Allocate a tensor on this backend's device
    fn alloc_tensor(
        &self,
        name: &str,
        shape: Vec<usize>,
        dtype: DataType,
    ) -> BackendResult<Tensor>;

    /// Allocate a tensor initialized to zero
    fn alloc_zeros(
        &self,
        name: &str,
        shape: Vec<usize>,
        dtype: DataType,
    ) -> BackendResult<Tensor> {
        let mut tensor = self.alloc_tensor(name, shape, dtype)?;
        self.fill_tensor(&mut tensor, 0.0)?;
        Ok(tensor)
    }

    /// Fill a tensor with a value
    fn fill_tensor(&self, tensor: &mut Tensor, value: f32)
        -> BackendResult<()>;

    /// Copy data to a tensor from CPU memory
    fn copy_to_tensor(
        &self,
        tensor: &mut Tensor,
        data: &[u8],
    ) -> BackendResult<()>;

    /// Copy data from a tensor to CPU memory
    fn copy_from_tensor(
        &self,
        tensor: &Tensor,
        data: &mut [u8],
    ) -> BackendResult<()>;

    /// Execute a compute graph
    fn execute(&self, graph: &mut ComputeGraph) -> BackendResult<()>;

    /// Synchronize (wait for all operations to complete)
    fn synchronize(&self) -> BackendResult<()>;

    /// Check if an operation is supported
    fn supports_op(&self, op_type: &super::op::OpType) -> bool;

    /// Check if a data type is supported
    fn supports_dtype(&self, dtype: &DataType) -> bool;
}

/// Backend factory
pub struct BackendFactory;

impl BackendFactory {
    /// Create a backend for the specified device
    pub fn create(device: Device) -> BackendResult<Box<dyn Backend>> {
        match device {
            Device::Cpu => Ok(Box::new(super::cpu_backend::CpuBackend::new())),
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            Device::Metal => {
                Ok(Box::new(super::metal_backend::MetalBackend::new()?))
            }
        }
    }
}
