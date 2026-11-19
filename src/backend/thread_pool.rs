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
/// Based on llama.cpp's thread allocation strategy:
/// - Decode (n=1): memory-bound GEMV, limit threads for cache efficiency
/// - Prefill (n>=8): compute-bound GEMM, use more threads
///
/// # Arguments
/// * `m` - Output dimension (rows)
/// * `n` - Batch size
/// * `k` - Input dimension (cols)
///
/// # Returns
/// Optimal number of threads
pub fn optimal_threads(m: usize, n: usize, k: usize) -> usize {
    let n_cpus = num_cpus();

    // Decode (n=1): memory-bound GEMV
    // Limit threads for cache efficiency
    if n == 1 {
        if m < 4096 {
            // Small decode: use fewer threads to avoid cache thrashing
            return 4.min(n_cpus);
        } else if m < 8192 {
            return 6.min(n_cpus);
        } else {
            return 8.min(n_cpus);
        }
    }

    // Prefill (n>=8): compute-bound GEMM
    // Use more threads with 2D tiling
    let flops = m.saturating_mul(n).saturating_mul(k);
    if flops < 10_000_000 {
        // Small matrix: fewer threads
        return 4.min(n_cpus);
    } else if flops < 100_000_000 {
        return n_cpus.min(6);
    } else {
        // Large matrix: use all threads
        return n_cpus;
    }
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
        // Small decode: 4 threads max
        assert_eq!(optimal_threads(1024, 1, 1024), 4);

        // Large decode: 8 threads max
        assert!(optimal_threads(8192, 1, 8192) >= 8);

        // Small prefill: 4 threads
        assert_eq!(optimal_threads(512, 8, 512), 4);

        // Large prefill: use all CPUs
        let n_cpus = num_cpus();
        assert_eq!(optimal_threads(4096, 32, 4096), n_cpus);
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
