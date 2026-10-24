// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::Buffer;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::rc::Rc;

/// Buffer location in heterogeneous memory hierarchy
///
/// Supports single-GPU inference today, extensible to heterogeneous architectures.
#[derive(Clone)]
pub enum BufferLocation {
    /// Local GPU VRAM (fast access, limited capacity)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    Local(Rc<Buffer>),

    /// Remote storage (AI Max or network, large capacity, slower access)
    /// Future: will hold RemoteHandle for transfer queue
    Remote {
        size: usize,
        /// Offset in remote storage (for now, just tracking metadata)
        offset: u64,
    },

    /// Host CPU memory (for CPU fallback operations)
    Host(Box<[u8]>),

    /// Buffer not yet loaded (pending transfer)
    Pending {
        size: usize,
        /// Source location (for transfer queue)
        source: Option<Box<BufferLocation>>,
    },
}

impl BufferLocation {
    /// Get buffer size in bytes
    pub fn size(&self) -> usize {
        match self {
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            Self::Local(buf) => buf.length() as usize,
            Self::Remote { size, .. } => *size,
            Self::Host(data) => data.len(),
            Self::Pending { size, .. } => *size,
        }
    }

    /// Check if buffer is in local GPU memory
    pub fn is_local(&self) -> bool {
        match self {
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            Self::Local(_) => true,
            _ => false,
        }
    }

    /// Check if buffer is on remote storage
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Remote { .. })
    }

    /// Check if buffer is in host memory
    pub fn is_host(&self) -> bool {
        matches!(self, Self::Host(_))
    }
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl BufferLocation {
    /// Get local GPU buffer if available
    pub fn as_local(&self) -> Option<&Rc<Buffer>> {
        match self {
            Self::Local(buf) => Some(buf),
            _ => None,
        }
    }
}
