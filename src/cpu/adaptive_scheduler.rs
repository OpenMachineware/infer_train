// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
// Adaptive scheduler for CPU matmul operations
// Inspired by llama.cpp's dynamic scheduling strategy

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Matmul execution path selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulPath {
    /// Single-threaded execution for small matrices
    /// Used when thread overhead exceeds compute benefit
    SingleThread,

    /// Tiled execution for batch >= 8
    /// Matches llama.cpp's tiled path threshold
    Tiled,

    /// Standard parallel execution
    /// Row-parallel with dynamic chunking
    Parallel,
}

/// Adaptive scheduler for CPU operations
///
/// Key design principles from llama.cpp:
/// 1. Dynamic thread count based on matrix dimensions
/// 2. Path selection based on batch size
/// 3. Dynamic chunk size to ensure load balancing
/// 4. Work-stealing for heterogeneous workloads
pub struct AdaptiveScheduler {
    /// Maximum available threads
    max_threads: usize,

    /// Work-stealing state (shared across threads)
    current_chunk: Arc<AtomicUsize>,
}

impl AdaptiveScheduler {
    pub fn new(max_threads: usize) -> Self {
        Self { max_threads, current_chunk: Arc::new(AtomicUsize::new(0)) }
    }

    /// Create with default thread count
    pub fn default_scheduler() -> Self {
        Self::new(crate::cpu::default_n_threads())
    }

    /// Compute optimal thread count for matmul
    ///
    /// From llama.cpp's chunking strategy (ggml-cpu.c:1465-1485):
    /// - Compute chunk_size (16 for GEMM, 64 for GEMV)
    /// - Calculate nchunk0 and nchunk1
    /// - If chunks < nth * 4, redistribute based on dimension size
    /// - When redistributing: larger dimension gets nth chunks, smaller gets 1
    pub fn optimal_threads(&self, m: usize, _k: usize, n: usize) -> usize {
        // Very small workload: single thread
        if m * n < 16 {
            return 1;
        }

        // Calculate chunking based on llama.cpp's strategy
        let chunk_size = if n == 1 { 64 } else { 16 };

        let nr0 = m;
        let nr1 = n;

        let mut nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        let mut nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        // llama.cpp: if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa())
        if nchunk0 * nchunk1 < self.max_threads * 4 {
            // Redistribute: larger dimension gets all threads
            if nr0 > nr1 {
                nchunk0 = self.max_threads;
                nchunk1 = 1;
            } else {
                nchunk0 = 1;
                nchunk1 = self.max_threads;
            }
        }

        // Thread count is the number of chunks in the primary dimension
        nchunk0.max(nchunk1).min(self.max_threads)
    }

    /// Select execution path based on batch size and dimensions
    ///
    /// llama.cpp threshold: batch >= 8 for tiled path
    pub fn select_path(
        &self,
        batch_size: usize,
        m: usize,
        k: usize,
    ) -> MatmulPath {
        // Single-thread for very small workloads
        if batch_size == 1 && m * k < 64 * 1024 {
            return MatmulPath::SingleThread;
        }

        // Tiled path for batch >= 8 (llama.cpp's threshold)
        if batch_size >= 8 {
            return MatmulPath::Tiled;
        }

        // Standard parallel for medium batch sizes
        if batch_size > 1 {
            return MatmulPath::Parallel;
        }

        // Decode path (batch = 1)
        // Use tiled even for batch=1 on decode (our optimization)
        MatmulPath::SingleThread
    }

    /// Compute optimal chunk size for parallel execution
    ///
    /// From llama.cpp's dynamic chunk sizing:
    /// - Ensure at least n_threads * 4 chunks for load balancing
    /// - Start from 256, reduce until enough chunks
    /// - Minimum chunk size is 16 (microtile size)
    pub fn compute_chunk_size(
        &self,
        rows: usize,
        cols: usize,
        n_threads: usize,
    ) -> usize {
        if n_threads == 1 {
            return rows; // Single thread: process all rows
        }

        let mut chunk_size = 256; // Start from TILED_TILE_ROWS

        // Compute number of chunks
        let mut nchunks = ((rows + chunk_size - 1) / chunk_size)
            * ((cols + chunk_size - 1) / chunk_size);

        // Reduce chunk size until we have enough chunks
        // Target: at least n_threads * 4 chunks for work-stealing
        while nchunks < n_threads * 4 && chunk_size > 16 {
            chunk_size /= 2;
            nchunks = ((rows + chunk_size - 1) / chunk_size)
                * ((cols + chunk_size - 1) / chunk_size);
        }

        chunk_size.max(16) // Minimum is microtile size
    }

    /// Initialize work-stealing for a new operation
    pub fn reset_work_stealing(&self) {
        self.current_chunk.store(0, Ordering::SeqCst);
    }

    /// Get next chunk for work-stealing
    /// Returns chunk index, or None if all chunks are taken
    pub fn get_next_chunk(&self, total_chunks: usize) -> Option<usize> {
        loop {
            let chunk = self.current_chunk.load(Ordering::SeqCst);
            if chunk >= total_chunks {
                return None;
            }

            // Try to claim this chunk
            match self.current_chunk.compare_exchange_weak(
                chunk,
                chunk + 1,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return Some(chunk),
                Err(_) => continue, // Another thread got it, retry
            }
        }
    }

    /// Compute chunk range for given chunk index
    pub fn get_chunk_range(
        &self,
        total_size: usize,
        chunk_size: usize,
        chunk_idx: usize,
    ) -> (usize, usize) {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(total_size);
        (start, end)
    }
}

// Thread-local scheduler instance
// Each thread has its own view of the shared work-stealing state
thread_local! {
    static SCHEDULER: AdaptiveScheduler = AdaptiveScheduler::default_scheduler();
}

/// Get the thread-local scheduler
pub fn get_scheduler() -> &'static AdaptiveScheduler {
    SCHEDULER.with(|s| unsafe {
        // This is safe because thread-local storage persists for the entire thread lifetime
        std::mem::transmute::<&AdaptiveScheduler, &'static AdaptiveScheduler>(
            &*s,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_threads_small_matrix() {
        let scheduler = AdaptiveScheduler::new(8);

        // Very small matrix: single thread
        assert_eq!(scheduler.optimal_threads(4, 128, 1), 1);

        // Medium matrix: all threads after redistribution
        // nchunk0 = 8 < 32, so redistribute: nchunk0 = 8, nchunk1 = 1
        assert_eq!(scheduler.optimal_threads(512, 512, 1), 8);

        // GEMV with large m: all threads
        // nchunk0 = 32 >= 32, so use all threads
        assert_eq!(scheduler.optimal_threads(2048, 4096, 1), 8);

        // Large matrix: all threads
        assert_eq!(scheduler.optimal_threads(8192, 4096, 1), 8);
    }

    #[test]
    fn test_path_selection() {
        let scheduler = AdaptiveScheduler::new(8);

        // Batch >= 8: tiled
        assert_eq!(scheduler.select_path(8, 1024, 1024), MatmulPath::Tiled);
        assert_eq!(scheduler.select_path(32, 1024, 1024), MatmulPath::Tiled);

        // Batch 2-7: parallel
        assert_eq!(scheduler.select_path(2, 1024, 1024), MatmulPath::Parallel);

        // Batch 1: single thread (decode)
        assert_eq!(
            scheduler.select_path(1, 1024, 1024),
            MatmulPath::SingleThread
        );
    }

    #[test]
    fn test_chunk_size() {
        let scheduler = AdaptiveScheduler::new(8);

        // Enough rows: use 256
        let chunk = scheduler.compute_chunk_size(8192, 128, 8);
        assert_eq!(chunk, 256);

        // Not enough rows: reduce chunk size
        let chunk = scheduler.compute_chunk_size(512, 128, 8);
        assert!(chunk < 256);
        assert!(chunk >= 16);

        // Single thread: process all
        let chunk = scheduler.compute_chunk_size(1024, 128, 1);
        assert_eq!(chunk, 1024);
    }

    #[test]
    fn test_work_stealing() {
        let scheduler = AdaptiveScheduler::new(4);
        scheduler.reset_work_stealing();

        // Get chunks sequentially
        let c0 = scheduler.get_next_chunk(10);
        let c1 = scheduler.get_next_chunk(10);
        let c2 = scheduler.get_next_chunk(10);

        assert_eq!(c0, Some(0));
        assert_eq!(c1, Some(1));
        assert_eq!(c2, Some(2));

        // Exhaust all chunks
        for _ in 3..10 {
            scheduler.get_next_chunk(10);
        }

        // Should return None
        let c = scheduler.get_next_chunk(10);
        assert_eq!(c, None);
    }
}
