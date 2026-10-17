// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use std::cell::UnsafeCell;
use std::rc::Rc;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Buffer;

/// Unified buffer reference
///
/// Provides a single interface for accessing buffers regardless of location.
/// Used by OperationExecutor to handle GPU and CPU buffers uniformly.
///
/// IMPORTANT: Returns Rc<> to ensure the same buffer is returned for multiple
/// get_or_load_on() calls with the same BufferId. This allows data written to
/// the buffer to persist across calls.
///
/// CPU buffers use UnsafeCell for interior mutability, allowing modification
/// through multiple Rc references. This is safe because BufferManager ensures
/// exclusive access through &mut self.
#[derive(Clone)]
pub enum BufferRef {
    /// GPU buffer (Metal)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    Gpu(Rc<Buffer>),

    /// CPU buffer (host memory)
    /// Wrapped in UnsafeCell for interior mutability.
    /// Safe because BufferManager ensures exclusive access.
    Cpu(Rc<UnsafeCell<Box<[u8]>>>),
}

impl BufferRef {
    /// Get buffer size in bytes
    pub fn size(&self) -> usize {
        match self {
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            Self::Gpu(buf) => buf.length() as usize,
            Self::Cpu(data) => unsafe { (&(*data.get())).len() },
        }
    }

    /// Check if buffer is in GPU memory
    pub fn is_gpu(&self) -> bool {
        match self {
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            Self::Gpu(_) => true,
            _ => false,
        }
    }

    /// Check if buffer is in CPU memory
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu(_))
    }
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl BufferRef {
    /// Get GPU buffer if available
    pub fn as_gpu(&self) -> Option<&Rc<Buffer>> {
        match self {
            Self::Gpu(buf) => Some(buf),
            _ => None,
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl BufferRef {
    /// Get CPU buffer if available (Metal builds - may be GPU buffer)
    pub fn as_cpu(&self) -> Option<&Box<[u8]>> {
        match self {
            Self::Cpu(data) => unsafe { Some(&*data.get()) },
            _ => None,
        }
    }

    /// Get CPU buffer as mutable slice (if CPU)
    ///
    /// Note: This is safe because BufferManager's &mut self ensures
    /// exclusive access to the buffer. Only one caller can have a
    /// BufferRef at a time through the BufferManager API.
    pub fn as_cpu_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Self::Cpu(data) => unsafe { Some((*data.get()).as_mut()) },
            _ => None,
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
impl BufferRef {
    /// Get CPU buffer (CPU-only builds - always available)
    pub fn as_cpu(&self) -> &Box<[u8]> {
        match self {
            Self::Cpu(data) => unsafe { &*data.get() },
        }
    }

    /// Get CPU buffer as mutable slice
    ///
    /// Note: This is safe because BufferManager's &mut self ensures
    /// exclusive access to the buffer. Only one caller can have a
    /// BufferRef at a time through the BufferManager API.
    pub fn as_cpu_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Cpu(data) => unsafe { (*data.get()).as_mut() },
        }
    }
}
