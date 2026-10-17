// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use std::collections::HashMap;

/// CPU buffer pool for reusing host memory
///
/// Similar to GPU BufferPool, but for CPU memory.
/// Used for:
/// - Pure CPU inference
/// - CPU fallback for unsupported GPU formats
/// - Temporary buffers during GPU → CPU → GPU transfers
pub struct BufferPoolHost {
    /// Pool organized by size class (rounded to power of 2)
    pool: HashMap<usize, Vec<Box<[u8]>>>,
    /// Total buffers created (for stats)
    total_created: usize,
    /// Buffers currently in use
    in_use: usize,
}

impl BufferPoolHost {
    pub fn new() -> Self {
        Self { pool: HashMap::new(), total_created: 0, in_use: 0 }
    }

    /// Get a buffer of the specified size (reuse if available)
    pub fn get(&mut self, size: usize) -> Box<[u8]> {
        let size_class = size.next_power_of_two();

        if let Some(buffers) = self.pool.get_mut(&size_class) {
            if let Some(buf) = buffers.pop() {
                self.in_use += 1;
                return buf;
            }
        }

        // Need to create a new buffer
        let buf = vec![0u8; size_class].into_boxed_slice();
        self.total_created += 1;
        self.in_use += 1;
        buf
    }

    /// Return a buffer to the pool for reuse
    pub fn release(&mut self, buf: Box<[u8]>) {
        let size_class = buf.len();
        self.pool.entry(size_class).or_insert_with(Vec::new).push(buf);
        self.in_use = self.in_use.saturating_sub(1);
    }

    /// Clear all cached buffers (free memory)
    pub fn clear(&mut self) {
        self.pool.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> BufferPoolHostStats {
        BufferPoolHostStats {
            total_created: self.total_created,
            in_use: self.in_use,
            cached: self.pool.values().map(|v| v.len()).sum(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BufferPoolHostStats {
    pub total_created: usize,
    pub in_use: usize,
    pub cached: usize,
}

impl Default for BufferPoolHost {
    fn default() -> Self {
        Self::new()
    }
}
