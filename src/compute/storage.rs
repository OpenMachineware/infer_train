// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Storage abstraction for cross-device tensors.

use std::sync::Arc;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Buffer;

/// Device type for tensor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    Metal,
}

/// Storage backend for tensor data.
#[derive(Clone, Debug)]
pub enum Storage {
    /// CPU heap storage.
    Cpu(Arc<Box<[u8]>>),
    /// Metal GPU buffer storage.
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    Metal(Arc<Buffer>),
}

impl Storage {
    /// Create CPU storage from bytes.
    pub fn cpu(data: Vec<u8>) -> Self {
        Storage::Cpu(Arc::new(data.into_boxed_slice()))
    }

    /// Get device type.
    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            Storage::Metal(_) => Device::Metal,
        }
    }

    /// Get raw byte slice (CPU only).
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Storage::Cpu(data) => Some(data.as_ref()),
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            Storage::Metal(_) => None,
        }
    }

    /// Get Metal buffer (Metal only).
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    pub fn as_metal_buffer(&self) -> Option<&Buffer> {
        match self {
            Storage::Cpu(_) => None,
            Storage::Metal(buffer) => Some(buffer),
        }
    }

    /// Get storage size in bytes.
    pub fn size(&self) -> usize {
        match self {
            Storage::Cpu(data) => data.len(),
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            Storage::Metal(buffer) => buffer.length() as usize,
        }
    }
}
