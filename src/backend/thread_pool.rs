// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// High-performance parallel execution for CPU inference
// Based on llama.cpp's thread allocation strategy with work stealing

use std::sync::atomic::{AtomicI32, Ordering};
use std::thread;

/// Get number of available CPU cores
fn num_cpus() -> usize {
    thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
}

/// Parallel execution with work stealing
///
/// Based on llama.cpp's ggml_threadpool design:
/// - Use atomic counter for work stealing (no locks)
/// - Cache-line aligned to avoid false sharing
/// - Optimal thread selection based on problem size
///
/// # Arguments
/// * `n_chunks` - Number of work chunks
/// * `n_threads` - Number of threads (0 = auto)
/// * `f` - Work function: f(chunk_id, thread_id)
///
/// # Example
/// ```
/// parallel_stealing(100, 8, |chunk, _tid| {
///     process_chunk(chunk);
/// });
/// ```
pub fn parallel_stealing<F>(n_chunks: usize, n_threads: usize, f: &F)
where
    F: Fn(usize, usize) + Send + Sync,
{
    if n_chunks == 0 {
        return;
    }

    let n_threads =
        if n_threads == 0 { num_cpus() } else { n_threads.min(num_cpus()) };

    if n_threads <= 1 || n_chunks <= 1 {
        // Single-threaded: no overhead
        for chunk in 0..n_chunks {
            f(chunk, 0);
        }
        return;
    }

    // Shared atomic counter for work stealing (cache-line aligned)
    #[repr(C, align(64))]
    struct AlignedCounter(AtomicI32);

    let counter = &AlignedCounter(AtomicI32::new(0));

    thread::scope(|s| {
        for tid in 0..n_threads {
            s.spawn(move || {
                loop {
                    // Atomically fetch next chunk
                    let chunk =
                        counter.0.fetch_add(1, Ordering::Relaxed) as usize;
                    if chunk >= n_chunks {
                        break;
                    }
                    f(chunk, tid);
                }
            });
        }
    });
}

/// Parallel execution with fixed distribution
///
/// Each thread gets a fixed range of chunks.
/// Better cache locality for memory-bound operations.
///
/// # Arguments
/// * `n_chunks` - Number of work chunks
/// * `n_threads` - Number of threads (0 = auto)
/// * `f` - Work function: f(chunk_start, chunk_end, thread_id)
///
/// # Performance Note
/// Thread creation overhead is ~109us per call on M1 Max.
/// For decode workloads (out_features < 10000), single-threaded is faster.
/// Use `PARALLEL_THRESHOLD` to tune this.
pub const PARALLEL_THRESHOLD: usize = 10000;

pub fn parallel_fixed<F>(n_chunks: usize, n_threads: usize, f: &F)
where
    F: Fn(usize, usize, usize) + Send + Sync,
{
    if n_chunks == 0 {
        return;
    }

    let n_threads =
        if n_threads == 0 { num_cpus() } else { n_threads.min(num_cpus()) };

    // CRITICAL: For small workloads, thread creation overhead (109us)
    // exceeds computation time. Use single-threaded.
    if n_threads <= 1 || n_chunks < PARALLEL_THRESHOLD {
        f(0, n_chunks, 0);
        return;
    }

    let chunk_size = (n_chunks + n_threads - 1) / n_threads;

    thread::scope(|s| {
        for tid in 0..n_threads {
            let start = tid * chunk_size;
            let end = (start + chunk_size).min(n_chunks);
            if start >= n_chunks {
                break;
            }
            s.spawn(move || {
                f(start, end, tid);
            });
        }
    });
}

/// Get optimal thread count for a given operation
///
/// Based on llama.cpp's tiled matmul strategy:
/// - Compute chunks: nchunk0 = ceil(m / chunk_size), nchunk1 = ceil(n / chunk_size)
/// - Use all threads when enough chunks exist
/// - Chunk size starts at 256, reduces to 16 minimum
///
/// NOTE: This returns optimal thread count for the *algorithm*, but actual
/// performance depends on using a persistent thread pool. Creating threads
/// on each call adds ~109us overhead, which can make multi-threaded decode
/// slower than single-threaded for small operations.
///
/// # Arguments
/// * `m` - Output dimension (rows)
/// * `n` - Batch size
/// * `k` - Input dimension (cols)
///
/// # Returns
/// Optimal number of threads
pub fn optimal_threads(m: usize, n: usize, _k: usize) -> usize {
    let n_cpus = num_cpus();

    // llama.cpp's chunking strategy:
    // Start with chunk_size=256, reduce until enough chunks
    let mut chunk_size = 256usize;
    let mut nchunk0 = (m + chunk_size - 1) / chunk_size;
    let mut nchunk1 = (n + chunk_size - 1) / chunk_size;

    // Reduce chunk_size to create enough chunks for all threads
    // Target: nchunk0 * nchunk1 >= n_cpus * 4
    while nchunk0 * nchunk1 < n_cpus * 4 && chunk_size > 16 {
        chunk_size /= 2;
        nchunk0 = (m + chunk_size - 1) / chunk_size;
        nchunk1 = (n + chunk_size - 1) / chunk_size;
    }

    // If we have enough chunks, use all CPUs
    // Otherwise, limit threads to available chunks
    let total_chunks = nchunk0 * nchunk1;
    total_chunks.min(n_cpus)
}

/// Get optimal thread count, capped at 1 for operations without persistent thread pool
///
/// Use this for operations that create threads on each call (like thread::scope).
/// Returns 1 to avoid thread creation overhead (~109us per call).
pub fn optimal_threads_no_pool(m: usize, n: usize, k: usize) -> usize {
    // For now, return 1 to avoid thread creation overhead
    // Once persistent thread pool is implemented, this can call optimal_threads()
    let _ = (m, n, k); // suppress unused warning
    1
}

/// Barrier synchronization using spin wait
///
/// Based on llama.cpp's ggml_barrier:
/// - No locks, pure atomic operations
/// - Spin wait with CPU relax
/// - Cache-line aligned to avoid false sharing
#[repr(C, align(64))]
pub struct SpinBarrier {
    n_threads: usize,
    counter: AtomicI32,
    generation: AtomicI32,
}

impl SpinBarrier {
    pub fn new(n_threads: usize) -> Self {
        Self {
            n_threads,
            counter: AtomicI32::new(0),
            generation: AtomicI32::new(0),
        }
    }

    /// Wait for all threads to reach the barrier
    pub fn wait(&self) {
        if self.n_threads <= 1 {
            return;
        }

        let current_gen = self.generation.load(Ordering::Relaxed);

        // Enter barrier
        let count = self.counter.fetch_add(1, Ordering::SeqCst);

        if count == (self.n_threads - 1) as i32 {
            // Last thread: reset and advance generation
            self.counter.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::SeqCst);
        } else {
            // Wait for generation to advance
            while self.generation.load(Ordering::Relaxed) == current_gen {
                std::hint::spin_loop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI32;

    #[test]
    fn test_parallel_stealing() {
        let counter = AtomicI32::new(0);
        parallel_stealing(100, 4, &|chunk, _tid| {
            if chunk < 100 {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(counter.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_parallel_fixed() {
        let counter = AtomicI32::new(0);
        parallel_fixed(100, 4, &|start, end, _tid| {
            for _chunk in start..end {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(counter.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_spin_barrier() {
        let barrier = SpinBarrier::new(4);
        let counter = AtomicI32::new(0);

        thread::scope(|s| {
            for _tid in 0..4 {
                s.spawn(|| {
                    counter.fetch_add(1, Ordering::Relaxed);
                    barrier.wait();
                    // All threads should have incremented
                    assert_eq!(counter.load(Ordering::Relaxed), 4);
                });
            }
        });
    }

    #[test]
    fn test_optimal_threads() {
        let n_cpus = num_cpus();

        // Decode with m=1024, n=1:
        // chunk_size reduces from 256 to 16 to create enough chunks
        // nchunk0 = ceil(1024/16) = 64 chunks
        // Use all available CPUs
        assert_eq!(optimal_threads(1024, 1, 1024), n_cpus);

        // Decode with m=8192, n=1: even more chunks
        assert_eq!(optimal_threads(8192, 1, 8192), n_cpus);

        // Prefill with m=512, n=8: many chunks from both dimensions
        assert_eq!(optimal_threads(512, 8, 512), n_cpus);

        // Large prefill: use all CPUs
        assert_eq!(optimal_threads(4096, 32, 4096), n_cpus);

        // Very small output dimension: limited chunks
        // m=64, chunk_size=16, nchunk0=4, nchunk1=1 (for n=1)
        // total=4 < n_cpus*4=40, but chunk_size already at minimum
        // So return min(4, n_cpus) = 4
        assert_eq!(optimal_threads(64, 1, 1024), 4.min(n_cpus));
    }

    #[test]
    fn test_parallel_stealing_single_chunk() {
        // Single chunk should work
        use std::sync::atomic::AtomicBool;
        let called = AtomicBool::new(false);
        parallel_stealing(1, 4, &|_chunk, _tid| {
            called.store(true, Ordering::Relaxed);
        });
        assert!(called.load(Ordering::Relaxed));
    }

    #[test]
    fn test_parallel_fixed_single_chunk() {
        // Single chunk should work
        use std::sync::atomic::AtomicBool;
        let called = AtomicBool::new(false);
        parallel_fixed(1, 4, &|_start, _end, _tid| {
            called.store(true, Ordering::Relaxed);
        });
        assert!(called.load(Ordering::Relaxed));
    }
}
