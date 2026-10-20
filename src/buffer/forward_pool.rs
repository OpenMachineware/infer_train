// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use std::cell::RefCell;
use std::rc::Rc;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::{Buffer, Device, MTLResourceOptions};

/// Buffer pool for a single forward pass
///
/// Keeps all intermediate buffers alive during command buffer encoding.
/// When the forward pass completes, buffers are returned to the reuse pool.
///
/// # Problem
/// Metal's `set_buffer()` doesn't retain buffer references. If a buffer's
/// last Rc drops before the command buffer commits, the buffer is deallocated
/// while the GPU still references it, causing crashes or zero outputs.
///
/// # Solution
/// This struct accumulates all Rc<Buffer> references during encoding.
/// The caller keeps this struct alive until the command buffer completes,
/// then drops it to release the buffers back to the reuse pool.
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
pub struct ForwardBufferPool {
    /// Device for creating new buffers
    device: Device,

    /// All buffers created during this forward pass
    /// We hold Rc references to prevent premature deallocation
    buffers: Vec<Rc<Buffer>>,

    /// Reuse pool for buffers (shared across forward passes)
    /// Each entry is (size_class, buffer)
    reuse_pool: Rc<RefCell<Vec<Rc<Buffer>>>>,
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl ForwardBufferPool {
    /// Create a new forward buffer pool
    ///
    /// # Arguments
    /// * `device` - Metal device for creating buffers
    /// * `reuse_pool` - Shared reuse pool for buffers (can be empty)
    pub fn new(
        device: Device,
        reuse_pool: Rc<RefCell<Vec<Rc<Buffer>>>>,
    ) -> Self {
        Self { device, buffers: Vec::new(), reuse_pool }
    }

    /// Get a buffer of the specified size
    ///
    /// Creates new buffer or reuses from pool. Does NOT zero the buffer.
    /// The buffer is kept alive in this pool until the forward pass completes.
    ///
    /// NOTE: Callers must ensure buffers are properly initialized by the GPU kernel.
    /// This avoids the massive overhead of zeroing large buffers on every forward pass.
    pub fn get(&mut self, size: usize) -> Rc<Buffer> {
        // Round up to power of 2 for better reuse
        let size_class = size.next_power_of_two();

        // Try to reuse from pool
        {
            let mut pool = self.reuse_pool.borrow_mut();
            if let Some(idx) =
                pool.iter().position(|b| b.length() as usize >= size_class)
            {
                let buf = pool.remove(idx);
                // DO NOT zero - GPU kernels will overwrite the data
                self.buffers.push(buf.clone());
                return buf;
            }
        }

        // Create new buffer (Metal doesn't zero new buffers)
        let buf = Rc::new(self.device.new_buffer(
            size_class as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        // DO NOT zero - GPU kernels will overwrite the data
        self.buffers.push(buf.clone());
        buf
    }

    /// Get a zeroed buffer of the specified size
    ///
    /// Same as `get()` but zeros the buffer contents.
    pub fn get_zeroed(&mut self, size: usize) -> Rc<Buffer> {
        let buf = self.get(size);
        unsafe {
            std::ptr::write_bytes(
                buf.contents() as *mut u8,
                0,
                buf.length() as usize,
            );
        }
        buf
    }

    /// Register an existing buffer to keep it alive
    ///
    /// Use this when you have a buffer that needs to stay alive
    /// for the duration of the forward pass.
    pub fn register(&mut self, buf: Rc<Buffer>) -> Rc<Buffer> {
        self.buffers.push(buf.clone());
        buf
    }

    /// Release all buffers to the reuse pool
    ///
    /// Call this after the command buffer completes.
    /// Buffers are returned to the reuse pool for future forward passes.
    ///
    /// IMPORTANT: Only returns buffers with reference count of 1.
    /// If a buffer is still referenced elsewhere (e.g., as layer output),
    /// it must not be reused to avoid data corruption.
    pub fn release_all(&mut self) {
        // Take ownership of the buffer list, leaving an empty Vec
        let buffers = std::mem::take(&mut self.buffers);

        for buf in buffers {
            // Now the only reference is 'buf' (if strong_count == 1)
            // If strong_count > 1, the buffer is still in use elsewhere
            if Rc::strong_count(&buf) == 1 {
                self.reuse_pool.borrow_mut().push(buf);
            }
            // Otherwise, let the buffer deallocate when 'buf' goes out of scope
        }
    }

    /// Get statistics
    pub fn stats(&self) -> ForwardBufferPoolStats {
        ForwardBufferPoolStats {
            active: self.buffers.len(),
            reusable: self.reuse_pool.borrow().len(),
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
impl Drop for ForwardBufferPool {
    fn drop(&mut self) {
        // Automatically return buffers to reuse pool
        self.release_all();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ForwardBufferPoolStats {
    pub active: usize,
    pub reusable: usize,
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
pub struct ForwardBufferPool;

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
impl ForwardBufferPool {
    pub fn new() -> Self {
        Self
    }

    pub fn stats(&self) -> ForwardBufferPoolStats {
        ForwardBufferPoolStats { active: 0, reusable: 0 }
    }
}
