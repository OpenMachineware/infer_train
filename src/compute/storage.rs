// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Storage abstraction for cross-device tensor data

use std::sync::Arc;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Buffer;

use super::device::Device;

/// Storage trait for tensor data on different devices
/// Note: We use StorageEnum instead of dyn Storage to avoid trait object overhead
pub trait StorageBase: std::fmt::Debug {
    /// Get the device this storage is on
    fn device(&self) -> Device;

    /// Get the size in bytes
    fn size(&self) -> usize;
}

/// CPU storage using reference-counted byte buffer
#[derive(Debug)]
pub struct CpuStorage {
    pub data: Arc<[u8]>,
}

impl CpuStorage {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data: data.into() }
    }

    pub fn from_arc(data: Arc<[u8]>) -> Self {
        Self { data }
    }

    pub fn zeros(size: usize) -> Self {
        Self { data: vec![0u8; size].into() }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn as_slice<T: bytemuck::Pod>(&self) -> Option<&[T]> {
        bytemuck::cast_slice(&self.data).into()
    }
}

impl StorageBase for CpuStorage {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn size(&self) -> usize {
        self.data.len()
    }
}

/// Metal storage using Metal buffer
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
#[derive(Debug)]
pub struct MetalStorage {
    buffer: std::rc::Rc<Buffer>,
    size: usize,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl MetalStorage {
    pub fn new(buffer: std::rc::Rc<Buffer>, size: usize) -> Self {
        Self { buffer, size }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn buffer_rc(&self) -> std::rc::Rc<Buffer> {
        self.buffer.clone()
    }
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl StorageBase for MetalStorage {
    fn device(&self) -> Device {
        Device::Metal
    }

    fn size(&self) -> usize {
        self.size
    }
}

/// Enum storage for convenience (avoids dyn trait overhead)
/// Note: MetalStorage uses Rc which is not thread-safe, so StorageEnum is not Send/Sync
#[derive(Debug, Clone)]
pub enum StorageEnum {
    Cpu(Arc<CpuStorage>),
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    Metal(Arc<MetalStorage>),
}

impl StorageEnum {
    pub fn device(&self) -> Device {
        match self {
            StorageEnum::Cpu(s) => s.device(),
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            StorageEnum::Metal(s) => s.device(),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            StorageEnum::Cpu(s) => s.size(),
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            StorageEnum::Metal(s) => s.size(),
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            StorageEnum::Cpu(s) => Some(s.data()),
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            StorageEnum::Metal(_) => None,
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn as_metal(&self) -> Option<&MetalStorage> {
        match self {
            StorageEnum::Metal(s) => Some(s),
            _ => None,
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, StorageEnum::Cpu(_))
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn is_metal(&self) -> bool {
        matches!(self, StorageEnum::Metal(_))
    }
}

/// Type alias for storage reference
pub type StorageRef = Arc<StorageEnum>;
