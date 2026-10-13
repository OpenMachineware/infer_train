// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Cross-device tensor abstraction

use std::sync::Arc;

use super::device::Device;
use super::storage::{CpuStorage, StorageEnum};
use crate::model::tensor::DataType;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use super::storage::MetalStorage;

/// Type alias for storage reference
pub type StorageRef = Arc<StorageEnum>;

/// A tensor that can exist on different devices
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Name for debugging and graph building
    pub name: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DataType,
    /// Storage (CPU, Metal, CUDA, etc.)
    pub storage: StorageRef,
}

impl Tensor {
    /// Create a new tensor with given storage
    pub fn new(
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: DataType,
        storage: StorageRef,
    ) -> Self {
        Self { name: name.into(), shape, dtype, storage }
    }

    /// Create a CPU tensor from data
    pub fn from_data(
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: DataType,
        data: Vec<u8>,
    ) -> Self {
        let storage =
            Arc::new(StorageEnum::Cpu(Arc::new(CpuStorage::new(data))));
        Self::new(name, shape, dtype, storage)
    }

    /// Create a CPU tensor from f32 data
    pub fn from_f32(
        name: impl Into<String>,
        shape: Vec<usize>,
        data: Vec<f32>,
    ) -> Self {
        let bytes = bytemuck::cast_slice(&data).to_vec();
        Self::from_data(name, shape, DataType::F32, bytes)
    }

    /// Create a zero CPU tensor
    pub fn zeros(
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: DataType,
    ) -> Self {
        let n_elems: usize = shape.iter().product();
        let n_bytes = dtype.bytes_for_elems(n_elems);
        let storage =
            Arc::new(StorageEnum::Cpu(Arc::new(CpuStorage::zeros(n_bytes))));
        Self::new(name, shape, dtype, storage)
    }

    /// Create a Metal tensor from an existing buffer
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn from_metal_buffer(
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: DataType,
        buffer: std::rc::Rc<metal::Buffer>,
    ) -> Self {
        let n_elems: usize = shape.iter().product();
        let n_bytes = dtype.bytes_for_elems(n_elems);
        let storage = Arc::new(StorageEnum::Metal(Arc::new(
            MetalStorage::new(buffer, n_bytes),
        )));
        Self::new(name, shape, dtype, storage)
    }

    /// Get device this tensor is on
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// Get number of elements
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get number of bytes
    pub fn byte_count(&self) -> usize {
        self.dtype.bytes_for_elems(self.elem_count())
    }

    /// Check if tensor is on CPU
    pub fn is_cpu(&self) -> bool {
        self.storage.is_cpu()
    }

    /// Check if tensor is on Metal
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn is_metal(&self) -> bool {
        self.storage.is_metal()
    }

    /// Get raw bytes (CPU only)
    pub fn as_bytes(&self) -> Option<&[u8]> {
        self.storage.as_bytes()
    }

    /// Get f32 slice (CPU F32 only)
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.dtype != DataType::F32 {
            return None;
        }
        self.as_bytes().map(|b| bytemuck::cast_slice(b))
    }

    /// Get mutable f32 slice (CPU F32 only, requires Arc uniqueness)
    /// Note: This only works if the tensor has a unique reference to its data.
    /// For most cases, you should create a new tensor instead of mutating.
    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        if self.dtype != DataType::F32 || !self.is_cpu() {
            return None;
        }

        // This won't work with Arc<[u8]>, so return None
        // Users should create a new tensor with from_f32 instead
        None
    }

    /// Get Metal buffer (Metal only)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn as_metal_buffer(&self) -> Option<std::rc::Rc<metal::Buffer>> {
        match self.storage.as_ref() {
            StorageEnum::Metal(metal_storage) => {
                Some(metal_storage.buffer_rc())
            }
            _ => None,
        }
    }

    /// Copy to CPU (if not already)
    pub fn to_cpu(&self) -> Self {
        if self.is_cpu() {
            return self.clone();
        }

        #[cfg(all(target_arch = "aarch64", feature = "metal"))]
        {
            if let Some(buffer) = self.as_metal_buffer() {
                let contents = buffer.contents();
                let data = unsafe {
                    std::slice::from_raw_parts(
                        contents as *const u8,
                        self.byte_count(),
                    )
                };
                return Self::from_data(
                    &self.name,
                    self.shape.clone(),
                    self.dtype,
                    data.to_vec(),
                );
            }
        }

        unimplemented!("to_cpu for this device")
    }

    /// Reshape tensor (no copy)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_elems: usize = new_shape.iter().product();
        assert_eq!(
            new_elems,
            self.elem_count(),
            "Reshape element count mismatch"
        );
        Self::new(
            self.name.clone(),
            new_shape,
            self.dtype,
            self.storage.clone(),
        )
    }
}

/// Trait for types that can be converted to/from tensor data
pub trait TensorElement:
    bytemuck::Pod + bytemuck::Zeroable + Copy + 'static
{
    const DTYPE: DataType;
}

impl TensorElement for f32 {
    const DTYPE: DataType = DataType::F32;
}
