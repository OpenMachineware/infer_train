// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Buffer Cache - Thread-local buffer pool for CPU inference
//
// Provides zero-allocation buffer access by caching pre-allocated buffers
// in thread-local storage. This eliminates the per-forward allocation overhead
// without changing operator function signatures.

use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// Cache of pre-allocated buffers, keyed by size
    /// Each entry is a Vec<f32> that can be reused
    static BUFFER_CACHE: RefCell<HashMap<usize, Vec<f32>>> = RefCell::new(HashMap::new());

    /// Statistics for tracking allocation patterns
    static STATS: RefCell<BufferStats> = RefCell::new(BufferStats::new());
}

#[derive(Debug, Default)]
struct BufferStats {
    total_requests: usize,
    cache_hits: usize,
    cache_misses: usize,
    allocations: usize,
    total_bytes_allocated: usize,
}

impl BufferStats {
    fn new() -> Self {
        BufferStats::default()
    }
}

/// Get a buffer of the specified size
///
/// If a buffer of this size exists in the cache, returns it.
/// Otherwise, allocates a new buffer and caches it.
///
/// This function returns a Vec<f32> that can be used directly.
/// The buffer is automatically returned to the cache when dropped.
#[inline]
pub fn get_buffer(size: usize) -> Vec<f32> {
    STATS.with(|stats_cell| {
        let mut stats = stats_cell.borrow_mut();
        stats.total_requests += 1;
    });

    BUFFER_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();

        // Try to find an existing buffer of the right size
        if let Some(mut buffer) = cache.remove(&size) {
            STATS.with(|stats_cell| {
                let mut stats = stats_cell.borrow_mut();
                stats.cache_hits += 1;
            });

            // Clear the buffer before returning
            for elem in buffer.iter_mut() {
                *elem = 0.0;
            }

            buffer
        } else {
            // Allocate a new buffer
            STATS.with(|stats_cell| {
                let mut stats = stats_cell.borrow_mut();
                stats.cache_misses += 1;
                stats.allocations += 1;
                stats.total_bytes_allocated +=
                    size * std::mem::size_of::<f32>();
            });

            vec![0.0f32; size]
        }
    })
}

/// Return a buffer to the cache
///
/// Call this after you're done with a buffer to enable reuse.
#[inline]
pub fn return_buffer(buffer: Vec<f32>) {
    let size = buffer.len();

    BUFFER_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();
        cache.insert(size, buffer);
    });
}

/// Get a buffer for temporary use (automatically returned when dropped)
///
/// This returns a Guard that automatically returns the buffer to the cache
/// when it goes out of scope.
pub fn get_temp_buffer(size: usize) -> BufferGuard {
    BufferGuard { buffer: Some(get_buffer(size)) }
}

/// A guard that automatically returns a buffer to the cache
pub struct BufferGuard {
    buffer: Option<Vec<f32>>,
}

impl BufferGuard {
    /// Get a reference to the buffer
    pub fn as_slice(&self) -> &[f32] {
        self.buffer.as_ref().unwrap()
    }

    /// Get a mutable reference to the buffer
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.buffer.as_mut().unwrap()
    }
}

impl Drop for BufferGuard {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            return_buffer(buffer);
        }
    }
}

/// Pre-allocate buffers for common sizes
///
/// Call this at the start of inference to pre-warm the cache.
pub fn preallocate_buffers(
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    q_size: usize,
    k_size: usize,
    v_size: usize,
    n_heads: usize,
    v_head_dim: usize,
    max_seq_len: usize,
) {
    BUFFER_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();

        // Pre-allocate buffers for each size
        let sizes = [
            hidden_size,
            intermediate_size,
            vocab_size,
            q_size,
            k_size,
            v_size,
            n_heads * v_head_dim,
            max_seq_len,
        ];

        // Allocate 2 copies for sizes that need double-buffering
        for &size in &sizes {
            cache.insert(size, vec![0.0f32; size]);
            if size == hidden_size || size == intermediate_size {
                cache.insert(size, vec![0.0f32; size]); // Second buffer for residual
            }
        }
    });

    println!("Pre-allocated buffers for CPU inference");
}

/// Clear the buffer cache
pub fn clear_cache() {
    BUFFER_CACHE.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();
        cache.clear();
    });
}

/// Get cache statistics
pub fn get_stats() -> (usize, usize, usize) {
    STATS.with(|stats_cell| {
        let stats = stats_cell.borrow();
        (stats.total_requests, stats.cache_hits, stats.cache_misses)
    })
}

/// Print cache statistics
pub fn print_stats() {
    STATS.with(|stats_cell| {
        let stats = stats_cell.borrow();
        println!(
            "Buffer cache stats: {} requests, {} hits, {} misses ({:.1}% hit rate)",
            stats.total_requests,
            stats.cache_hits,
            stats.cache_misses,
            if stats.total_requests > 0 {
                100.0 * stats.cache_hits as f32 / stats.total_requests as f32
            } else {
                0.0
            }
        );
        println!(
            "  Allocations: {}, Total bytes: {}",
            stats.allocations,
            stats.total_bytes_allocated
        );
    });
}
