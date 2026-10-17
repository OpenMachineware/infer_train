// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::rc::Rc;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Buffer;

/// Unified buffer reference
///
/// Provides a single interface for accessing buffers regardless of location.
/// Used by OperationExecutor to handle GPU and CPU buffers uniformly.
#[derive(Clone)]
pub enum BufferRef {
    /// GPU buffer (Metal)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    Gpu(Rc<Buffer>),

    /// CPU buffer (host memory)
    Cpu(Box<[u8]>),
}

impl BufferRef {
    /// Get buffer size in bytes
    pub fn size(&self) -> usize {
        match self {
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            Self::Gpu(buf) => buf.length() as usize,
            Self::Cpu(data) => data.len(),
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

impl BufferRef {
    /// Get CPU buffer if available
    pub fn as_cpu(&self) -> Option<&Box<[u8]>> {
        match self {
            Self::Cpu(data) => Some(data),
            _ => None,
        }
    }

    /// Get CPU buffer as mutable slice (if CPU)
    pub fn as_cpu_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Self::Cpu(data) => Some(data.as_mut()),
            _ => None,
        }
    }
}
