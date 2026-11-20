// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Compute arena for zero-allocation inference
//!
//! Based on llama.cpp's `ggml_cplan.work_data` design:
//! - Pre-allocate a single buffer at model load time
//! - Reset before each forward pass
//! - All temporary allocations reuse the same memory

use std::alloc::{Layout, alloc, dealloc};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Compute arena for temporary buffer allocation
///
/// # Safety
///
/// The arena provides raw memory that can be reused across forward passes.
/// Callers must ensure:
/// 1. `reset()` is called before each forward pass
/// 2. References from previous allocations are not used after reset
/// 3. Proper alignment is maintained when casting to different types
pub struct ComputeArena {
    /// Pre-allocated buffer
    buffer: *mut u8,
    /// Total buffer size in bytes
    capacity: usize,
    /// Current allocation offset (atomic for thread-safe reset)
    offset: AtomicUsize,
}

unsafe impl Send for ComputeArena {}
unsafe impl Sync for ComputeArena {}

impl ComputeArena {
    /// Create a new arena with the specified capacity in bytes
    pub fn new(capacity: usize) -> Self {
        let layout =
            Layout::from_size_align(capacity, 64).expect("Invalid layout");
        let buffer = unsafe { alloc(layout) };
        if buffer.is_null() {
            panic!("Failed to allocate arena buffer of {} bytes", capacity);
        }
        Self { buffer, capacity, offset: AtomicUsize::new(0) }
    }

    /// Reset the arena for a new forward pass
    ///
    /// # Safety
    ///
    /// After calling reset, all previously allocated memory is available for reuse.
    /// Do not use any references obtained before reset.
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Relaxed);
    }

    /// Allocate `count` elements of type T
    ///
    /// Returns a mutable slice of uninitialized memory.
    /// The memory is zeroed on first use if needed.
    pub fn alloc<T>(&self, count: usize) -> &mut [T] {
        let size = count * std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();

        // Align offset
        let current = self.offset.load(Ordering::Relaxed);
        let aligned = (current + align - 1) & !(align - 1);
        let new_offset = aligned + size;

        if new_offset > self.capacity {
            panic!(
                "Arena overflow: requested {} bytes, available {} bytes",
                size,
                self.capacity - current
            );
        }

        // Update offset
        self.offset.store(new_offset, Ordering::Relaxed);

        // Return slice
        unsafe {
            let ptr = self.buffer.add(aligned) as *mut T;
            std::slice::from_raw_parts_mut(ptr, count)
        }
    }

    /// Allocate and zero-initialize `count` elements of type T
    pub fn alloc_zeroed<T: Copy>(&self, count: usize) -> &mut [T] {
        let slice = self.alloc(count);
        unsafe {
            std::ptr::write_bytes(slice.as_mut_ptr(), 0, count);
        }
        slice
    }

    /// Allocate a single element
    pub fn alloc_one<T>(&self) -> &mut T {
        let slice = self.alloc(1);
        &mut slice[0]
    }

    /// Get current offset (for debugging)
    pub fn offset(&self) -> usize {
        self.offset.load(Ordering::Relaxed)
    }

    /// Get total capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get remaining capacity
    pub fn remaining(&self) -> usize {
        self.capacity - self.offset.load(Ordering::Relaxed)
    }
}

impl Drop for ComputeArena {
    fn drop(&mut self) {
        if !self.buffer.is_null() {
            let layout = Layout::from_size_align(self.capacity, 64)
                .expect("Invalid layout");
            unsafe {
                dealloc(self.buffer, layout);
            }
        }
    }
}

/// Calculate the maximum work buffer size needed for a model
///
/// This should be called once at model load time to determine
/// the arena capacity.
pub fn calculate_work_size(
    hidden_size: usize,
    intermediate_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    _n_layers: usize,
) -> usize {
    use std::mem::size_of;

    // Size of Q8_K block for quantized input
    let q8k_block_size = size_of::<crate::types::BlockQ8_K>();
    let qk_k = crate::types::QK_K;

    // Quantized input buffer (shared for all linear ops)
    let nb = (hidden_size + qk_k - 1) / qk_k;
    let quant_input_size = nb * q8k_block_size;

    // Attention stage buffers
    let q_size = n_heads * head_dim * size_of::<f32>();
    let k_size = n_kv_heads * head_dim * size_of::<f32>();
    let v_size = n_kv_heads * head_dim * size_of::<f32>();
    let q_normed_size = n_heads * head_dim * size_of::<f32>();
    let k_normed_size = n_kv_heads * head_dim * size_of::<f32>();
    let attn_output_size = n_heads * head_dim * size_of::<f32>();
    let attn_out_size = hidden_size * size_of::<f32>(); // output projection result

    // FFN stage buffers
    let gate_size = intermediate_size * size_of::<f32>();
    let up_size = intermediate_size * size_of::<f32>();
    let ffn_out_size = hidden_size * size_of::<f32>();

    // Output projection buffer
    let logits_size = vocab_size * size_of::<f32>();

    // Compute total memory usage within a single forward_layer_inplace call
    // All allocations accumulate since they're in the same function scope
    // Attention stage: normed_attn + q + k + v + q_normed + k_normed + attn_output + attn_out
    // FFN stage (added): normed_ffn + gate + up + ffn_out
    let total_layer = hidden_size * size_of::<f32>()  // normed_attn
        + q_size
        + k_size
        + v_size
        + q_normed_size
        + k_normed_size
        + attn_output_size
        + attn_out_size
        + hidden_size * size_of::<f32>()  // normed_ffn
        + gate_size
        + up_size
        + ffn_out_size;

    // Add quantized input buffer and logits
    // Add 20% safety margin for alignment
    let total = total_layer + quant_input_size + logits_size;
    (total * 120) / 100
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let arena = ComputeArena::new(1024);

        // Allocate some f32
        let slice = arena.alloc_zeroed::<f32>(10);
        assert_eq!(slice.len(), 10);
        assert!(slice.iter().all(|&x| x == 0.0));

        // Allocate more
        let slice2 = arena.alloc::<u8>(100);
        assert_eq!(slice2.len(), 100);

        assert!(arena.offset() <= 1024);
    }

    #[test]
    fn test_arena_reset() {
        let arena = ComputeArena::new(1024);

        let _ = arena.alloc::<f32>(10);
        assert!(arena.offset() > 0);

        arena.reset();
        assert_eq!(arena.offset(), 0);
    }

    #[test]
    fn test_qwen3_buffer_size() {
        // Qwen3-0.6B
        let size = calculate_work_size(1024, 3072, 16, 8, 128, 151936, 28);
        println!(
            "Qwen3-0.6B work buffer size: {} bytes ({:.2} MB)",
            size,
            size as f64 / 1024.0 / 1024.0
        );

        // Should be reasonable (< 10 MB)
        assert!(size < 10 * 1024 * 1024);
    }
}
