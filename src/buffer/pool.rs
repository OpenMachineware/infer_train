// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::collections::HashMap;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::rc::Rc;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::{Buffer, Device, MTLResourceOptions};

/// Buffer pool for reusing buffers across tokens
///
/// Instead of allocating new buffers for each operation, maintain a pool
/// of reusable buffers organized by size. This reduces allocation overhead
/// and memory fragmentation.
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct BufferPool {
    /// Pool organized by size class (rounded to power of 2)
    /// Each entry contains a list of available buffers
    pool: HashMap<usize, Vec<Rc<Buffer>>>,

    /// Device for creating new buffers
    device: Device,

    /// Total buffers created (for stats)
    total_created: usize,

    /// Buffers currently in use
    in_use: usize,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl BufferPool {
    pub fn new(device: Device) -> Self {
        Self { pool: HashMap::new(), device, total_created: 0, in_use: 0 }
    }

    /// Get a buffer of the specified size (reuse if available)
    ///
    /// Sizes are rounded up to the nearest power of 2 for better reuse.
    pub fn get(&mut self, size: usize) -> Rc<Buffer> {
        // Round up to power of 2 for better reuse
        let size_class = size.next_power_of_two();

        if let Some(buffers) = self.pool.get_mut(&size_class) {
            if let Some(buf) = buffers.pop() {
                self.in_use += 1;
                return buf;
            }
        }

        // Need to create a new buffer
        let buf = Rc::new(self.device.new_buffer(
            size_class as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        self.total_created += 1;
        self.in_use += 1;
        buf
    }

    /// Return a buffer to the pool for reuse
    pub fn release(&mut self, buf: Rc<Buffer>) {
        let size_class = buf.length() as usize;
        self.pool.entry(size_class).or_insert_with(Vec::new).push(buf);
        self.in_use = self.in_use.saturating_sub(1);
    }

    /// Clear all cached buffers (free memory)
    pub fn clear(&mut self) {
        self.pool.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> BufferPoolStats {
        BufferPoolStats {
            total_created: self.total_created,
            in_use: self.in_use,
            cached: self.pool.values().map(|v| v.len()).sum(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BufferPoolStats {
    pub total_created: usize,
    pub in_use: usize,
    pub cached: usize,
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
pub struct BufferPool;

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
impl BufferPool {
    pub fn new() -> Self {
        Self
    }

    pub fn stats(&self) -> BufferPoolStats {
        BufferPoolStats { total_created: 0, in_use: 0, cached: 0 }
    }
}
