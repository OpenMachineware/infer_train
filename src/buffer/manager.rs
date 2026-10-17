// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use std::collections::HashMap;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::rc::Rc;

use super::{BufferId, BufferLocation, BufferPool, BufferPoolHost, BufferRef};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::{Buffer, Device};

/// Compute device selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeDevice {
    /// Local GPU (Metal on macOS)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    LocalGpu,
    /// CPU (host memory)
    Cpu,
}

/// Buffer Manager (Layer 1)
///
/// Manages buffer locations and lifecycle across the three-layer architecture.
/// Supports single-GPU inference today, extensible to heterogeneous architectures.
///
/// Key responsibilities:
/// - Track where each buffer is located (Local GPU / Remote / Host)
/// - Manage buffer pool for reuse across tokens
/// - Support both GPU and CPU buffers for fallback execution
/// - Provide transfer queue interface (future: for heterogeneity)
/// - Prefetch support for expert scheduling
pub struct BufferManager {
    /// Registry of all known buffers by ID
    registry: HashMap<BufferId, BufferLocation>,

    /// Local cache of GPU buffers (for fast access)
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    local_cache: HashMap<BufferId, Rc<Buffer>>,

    /// Buffer pool for GPU reuse
    pool: BufferPool,

    /// CPU buffer pool for host memory reuse
    host_pool: BufferPoolHost,

    /// CPU buffer cache
    host_cache: HashMap<BufferId, Box<[u8]>>,

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
            host_pool: BufferPoolHost::new(),
            host_cache: HashMap::new(),
            pinned: HashMap::new(),
            device,
        }
    }

    /// Get a buffer on specific device (GPU or CPU)
    ///
    /// Primary interface for accessing buffers with device selection.
    pub fn get_or_load_on(
        &mut self,
        id: BufferId,
        size: usize,
        device: ComputeDevice,
    ) -> BufferRef {
        match device {
            ComputeDevice::LocalGpu => {
                let buf = self.get_or_load(id, size);
                BufferRef::Gpu(buf)
            }
            ComputeDevice::Cpu => {
                // Check if already in host cache
                if let Some(buf) = self.host_cache.get(&id) {
                    return BufferRef::Cpu(buf.clone());
                }

                // Check registry for existing location
                if let Some(location) = self.registry.get(&id) {
                    match location {
                        BufferLocation::Host(data) => {
                            // Already in host memory
                            self.host_cache.insert(id, data.clone());
                            return BufferRef::Cpu(data.clone());
                        }
                        #[cfg(all(target_arch = "aarch64", feature = "metal"))]
                        BufferLocation::Local(gpu_buf) => {
                            // Future: transfer from GPU to CPU
                            // For now, create new CPU buffer
                            let buf = self.host_pool.get(size);
                            self.host_cache.insert(id, buf.clone());
                            return BufferRef::Cpu(buf);
                        }
                        _ => {
                            // Create new CPU buffer
                            let buf = self.host_pool.get(size);
                            self.host_cache.insert(id, buf.clone());
                            return BufferRef::Cpu(buf);
                        }
                    }
                }

                // Buffer doesn't exist, create it
                let buf = self.host_pool.get(size);
                self.host_cache.insert(id, buf.clone());
                self.registry.insert(id, BufferLocation::Host(buf.clone()));
                BufferRef::Cpu(buf)
            }
        }
    }

    /// Get a GPU buffer, loading from remote if necessary
    ///
    /// Legacy interface for GPU-only access.
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

    /// Register a pre-existing GPU buffer (e.g., weight tensor)
    pub fn register(&mut self, id: BufferId, buffer: Rc<Buffer>) {
        self.registry.insert(id, BufferLocation::Local(buffer.clone()));
        self.local_cache.insert(id, buffer);
    }

    /// Register a pre-existing CPU buffer
    pub fn register_host(&mut self, id: BufferId, buffer: Box<[u8]>) {
        self.registry.insert(id, BufferLocation::Host(buffer.clone()));
        self.host_cache.insert(id, buffer);
    }

    /// Get current buffer location
    pub fn get_location(&self, id: &BufferId) -> Option<&BufferLocation> {
        self.registry.get(id)
    }

    /// Transfer buffer between devices
    ///
    /// Future: will implement actual data transfer.
    /// For now, just metadata tracking.
    pub fn transfer(
        &mut self,
        id: BufferId,
        from: ComputeDevice,
        to: ComputeDevice,
    ) {
        // Future: implement actual data transfer
        // GPU -> CPU: buffer.contents() copy
        // CPU -> GPU: buffer.contents() write
        // For now, just log the request
        #[cfg(debug_assertions)]
        eprintln!(
            "[BufferManager] Transfer requested: {:?} -> {:?} for buffer {:?}",
            from, to, id
        );
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
        // Evict GPU buffers
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

        // Evict CPU buffers
        let to_evict_host: Vec<BufferId> = self
            .host_cache
            .keys()
            .filter(|id| !self.pinned.contains_key(id))
            .cloned()
            .collect();

        for id in to_evict_host {
            if let Some(buf) = self.host_cache.remove(&id) {
                self.host_pool.release(buf);
                // Update registry
                if let Some(location) = self.registry.get_mut(&id) {
                    *location = BufferLocation::Remote { size: 0, offset: 0 };
                }
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> BufferManagerStats {
        BufferManagerStats {
            registered: self.registry.len(),
            cached_gpu: self.local_cache.len(),
            cached_cpu: self.host_cache.len(),
            pinned: self.pinned.len(),
            pool_gpu: self.pool.stats(),
            pool_cpu: self.host_pool.stats(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BufferManagerStats {
    pub registered: usize,
    pub cached_gpu: usize,
    pub cached_cpu: usize,
    pub pinned: usize,
    pub pool_gpu: super::pool::BufferPoolStats,
    pub pool_cpu: super::pool_host::BufferPoolHostStats,
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
impl BufferManager {
    pub fn new() -> Self {
        Self {
            registry: HashMap::new(),
            pool: BufferPool::new(),
            host_pool: BufferPoolHost::new(),
            host_cache: HashMap::new(),
            pinned: HashMap::new(),
        }
    }

    pub fn get_or_load_on(
        &mut self,
        id: BufferId,
        size: usize,
        device: ComputeDevice,
    ) -> BufferRef {
        match device {
            ComputeDevice::Cpu => {
                if let Some(buf) = self.host_cache.get(&id) {
                    return BufferRef::Cpu(buf.clone());
                }
                let buf = self.host_pool.get(size);
                self.host_cache.insert(id, buf.clone());
                self.registry.insert(id, BufferLocation::Host(buf.clone()));
                BufferRef::Cpu(buf)
            }
        }
    }

    pub fn get_location(&self, id: &BufferId) -> Option<&BufferLocation> {
        self.registry.get(id)
    }

    pub fn transfer(
        &mut self,
        _id: BufferId,
        _from: ComputeDevice,
        _to: ComputeDevice,
    ) {
        // No-op on non-Metal platforms
    }

    pub fn pin(&mut self, id: BufferId) {
        self.pinned.insert(id, true);
    }

    pub fn unpin(&mut self, id: BufferId) {
        self.pinned.remove(&id);
    }

    pub fn is_pinned(&self, id: &BufferId) -> bool {
        self.pinned.contains_key(id)
    }

    pub fn evict(&mut self) {
        // Evict CPU buffers
        let to_evict_host: Vec<BufferId> = self
            .host_cache
            .keys()
            .filter(|id| !self.pinned.contains_key(id))
            .cloned()
            .collect();

        for id in to_evict_host {
            if let Some(buf) = self.host_cache.remove(&id) {
                self.host_pool.release(buf);
                // Update registry
                if let Some(location) = self.registry.get_mut(&id) {
                    *location = BufferLocation::Remote { size: 0, offset: 0 };
                }
            }
        }
    }

    pub fn stats(&self) -> BufferManagerStats {
        BufferManagerStats {
            registered: self.registry.len(),
            cached_gpu: 0,
            cached_cpu: self.host_cache.len(),
            pinned: self.pinned.len(),
            pool_gpu: self.pool.stats(),
            pool_cpu: self.host_pool.stats(),
        }
    }
}
