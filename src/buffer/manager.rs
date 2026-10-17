// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use std::collections::HashMap;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::rc::Rc;

use super::{BufferId, BufferLocation, BufferPool};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::{Buffer, Device};

/// Buffer Manager (Layer 1)
///
/// Manages buffer locations and lifecycle across the three-layer architecture.
/// Supports single-GPU inference today, extensible to heterogeneous architectures.
///
/// Key responsibilities:
/// - Track where each buffer is located (Local GPU / Remote / Host)
/// - Manage buffer pool for reuse across tokens
/// - Provide transfer queue interface (future: for heterogeneity)
/// - Prefetch support for expert scheduling
pub struct BufferManager {
    /// Registry of all known buffers by ID
    registry: HashMap<BufferId, BufferLocation>,

    /// Local cache of GPU buffers (for fast access)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    local_cache: HashMap<BufferId, Rc<Buffer>>,

    /// Buffer pool for reuse
    pool: BufferPool,

    /// Pinned buffers (should not be evicted)
    /// Used for frequently accessed weights (e.g., embedding, output layer)
    pinned: HashMap<BufferId, bool>,

    /// Device for creating buffers
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    device: Device,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl BufferManager {
    pub fn new(device: Device) -> Self {
        Self {
            registry: HashMap::new(),
            local_cache: HashMap::new(),
            pool: BufferPool::new(device.clone()),
            pinned: HashMap::new(),
            device,
        }
    }

    /// Get a buffer, loading from remote if necessary
    ///
    /// This is the primary interface for accessing buffers.
    /// - If buffer is already local, return it
    /// - If buffer is remote, trigger transfer (future)
    /// - If buffer doesn't exist, create it
    pub fn get_or_load(&mut self, id: BufferId, size: usize) -> Rc<Buffer> {
        // Check if already in local cache
        if let Some(buf) = self.local_cache.get(&id) {
            return buf.clone();
        }

        // Check registry for location
        if let Some(location) = self.registry.get(&id) {
            match location {
                BufferLocation::Local(buf) => {
                    // Already local, cache it
                    self.local_cache.insert(id, buf.clone());
                    return buf.clone();
                }
                BufferLocation::Remote { .. } => {
                    // Future: trigger transfer from remote
                    // For now, create a new buffer
                    let buf = self.pool.get(size);
                    self.local_cache.insert(id, buf.clone());
                    return buf;
                }
                BufferLocation::Host(_) => {
                    // Future: copy from host to GPU
                    // For now, create a new buffer
                    let buf = self.pool.get(size);
                    self.local_cache.insert(id, buf.clone());
                    return buf;
                }
                BufferLocation::Pending { .. } => {
                    // Future: wait for pending transfer
                    // For now, create a new buffer
                    let buf = self.pool.get(size);
                    self.local_cache.insert(id, buf.clone());
                    return buf;
                }
            }
        }

        // Buffer doesn't exist, create it
        let buf = self.pool.get(size);
        self.local_cache.insert(id, buf.clone());
        self.registry.insert(id, BufferLocation::Local(buf.clone()));
        buf
    }

    /// Register a pre-existing buffer (e.g., weight tensor)
    pub fn register(&mut self, id: BufferId, buffer: Rc<Buffer>) {
        self.registry.insert(id, BufferLocation::Local(buffer.clone()));
        self.local_cache.insert(id, buffer);
    }

    /// Prefetch buffers (for expert scheduling)
    ///
    /// Trigger async transfer of buffers from remote to local cache.
    /// Future: will use transfer queue for async operation.
    pub fn prefetch(&mut self, _ids: &[BufferId]) {
        // Future: implement async transfer queue
        // For now, no-op
    }

    /// Pin a buffer (prevent eviction)
    ///
    /// Pinned buffers stay in local cache even when cache is full.
    /// Use for frequently accessed weights.
    pub fn pin(&mut self, id: BufferId) {
        self.pinned.insert(id, true);
    }

    /// Unpin a buffer (allow eviction)
    pub fn unpin(&mut self, id: BufferId) {
        self.pinned.remove(&id);
    }

    /// Check if a buffer is pinned
    pub fn is_pinned(&self, id: &BufferId) -> bool {
        self.pinned.contains_key(id)
    }

    /// Evict unpinned buffers from local cache
    ///
    /// Returns buffers to pool for reuse.
    /// Use when local cache is too large.
    pub fn evict(&mut self) {
        let to_evict: Vec<BufferId> = self
            .local_cache
            .keys()
            .filter(|id| !self.pinned.contains_key(id))
            .cloned()
            .collect();

        for id in to_evict {
            if let Some(buf) = self.local_cache.remove(&id) {
                self.pool.release(buf);
                // Update registry to reflect eviction
                if let Some(location) = self.registry.get_mut(&id) {
                    *location = BufferLocation::Remote {
                        size: 0, // Unknown after eviction
                        offset: 0,
                    };
                }
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> BufferManagerStats {
        BufferManagerStats {
            registered: self.registry.len(),
            cached: self.local_cache.len(),
            pinned: self.pinned.len(),
            pool: self.pool.stats(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BufferManagerStats {
    pub registered: usize,
    pub cached: usize,
    pub pinned: usize,
    pub pool: super::pool::BufferPoolStats,
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
impl BufferManager {
    pub fn new() -> Self {
        Self {
            registry: HashMap::new(),
            pool: BufferPool::new(),
            pinned: HashMap::new(),
        }
    }

    pub fn stats(&self) -> BufferManagerStats {
        BufferManagerStats {
            registered: self.registry.len(),
            cached: 0,
            pinned: self.pinned.len(),
            pool: self.pool.stats(),
        }
    }
}
