// SPDX-License-Identifier: AGPL-3.0-or-later
// Persistent thread pool for CPU inference

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};

/// Get the default number of threads to use
pub fn default_n_threads() -> usize {
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4)
}

/// Atomic spin barrier - matches llama.cpp's ggml_barrier
/// Uses two counters for proper reusability
pub struct SpinBarrier {
    n_barrier: AtomicUsize,
    n_barrier_passed: AtomicUsize,
    n_threads: usize,
}

// SpinBarrier is Send + Sync because it only contains atomic types
unsafe impl Send for SpinBarrier {}
unsafe impl Sync for SpinBarrier {}

impl SpinBarrier {
    pub fn new(n_threads: usize) -> Self {
        SpinBarrier {
            n_barrier: AtomicUsize::new(0),
            n_barrier_passed: AtomicUsize::new(0),
            n_threads,
        }
    }

    /// Wait for all threads to reach the barrier
    /// Uses spin loop for low latency - matches llama.cpp exactly
    pub fn wait(&self) {
        if self.n_threads == 1 {
            return;
        }

        let n_passed = self.n_barrier_passed.load(Ordering::Relaxed);

        // Enter barrier with seq-cst fence
        let n_barrier = self.n_barrier.fetch_add(1, Ordering::SeqCst);

        if n_barrier == (self.n_threads - 1) {
            // Last thread - reset and signal
            self.n_barrier.store(0, Ordering::Relaxed);
            self.n_barrier_passed.fetch_add(1, Ordering::SeqCst);
            return;
        }

        // Wait for other threads
        while self.n_barrier_passed.load(Ordering::Relaxed) == n_passed {
            std::hint::spin_loop();
        }

        // Exit barrier with seq-cst fence
        std::sync::atomic::fence(Ordering::SeqCst);
    }
}

/// Parallel chunks using work stealing with atomic counter
pub fn parallel_work_stealing<F>(
    n_items: usize,
    chunk_size: usize,
    n_threads: usize,
    f: F,
) where
    F: Fn(usize, usize) + Send + Sync,
{
    if n_items == 0 {
        return;
    }

    let counter = AtomicUsize::new(0);
    let f_ref = &f;

    std::thread::scope(|s| {
        for _ in 0..n_threads {
            s.spawn(|| {
                loop {
                    let start =
                        counter.fetch_add(chunk_size, Ordering::Relaxed);
                    if start >= n_items {
                        break;
                    }
                    let end = (start + chunk_size).min(n_items);
                    f_ref(start, end);
                }
            });
        }
    });
}

/// Parallel chunks with fixed distribution using std::thread::scope
/// Falls back to single-threaded only for n_threads == 1
pub fn parallel_chunks<F>(n_items: usize, n_threads: usize, f: F)
where
    F: Fn(usize, usize) + Send + Sync + Clone,
{
    if n_items == 0 {
        return;
    }

    // Single-threaded only if explicitly requested
    if n_threads == 1 {
        f(0, n_items);
        return;
    }

    let chunk_size = (n_items + n_threads - 1) / n_threads;

    std::thread::scope(|s| {
        for thread_id in 0..n_threads {
            let start = thread_id * chunk_size;
            let end = (start + chunk_size).min(n_items);

            if start >= n_items {
                break;
            }

            let f_clone = f.clone();
            s.spawn(move || {
                f_clone(start, end);
            });
        }
    });
}

/// Persistent thread pool - matches llama.cpp's ggml_threadpool exactly
///
/// Key design from llama.cpp:
/// 1. Threads wait on atomic "current_op" counter (spin loop)
/// 2. Main thread increments counter to wake all workers simultaneously
/// 3. Workers process op, then increment completion counter
/// 4. NO channel overhead - atomic spin for sub-microsecond wake
pub struct PersistentThreadPool {
    handles: Vec<JoinHandle<()>>,
    /// Current operation number - workers spin-wait until this changes
    current_op: Arc<AtomicUsize>,
    /// Function pointer stored in RwLock for concurrent reads
    work_fn:
        Arc<std::sync::RwLock<Option<Arc<dyn Fn(usize, usize) + Send + Sync>>>>,
    /// Number of items to process
    num_items: Arc<AtomicUsize>,
    /// Completion counter - workers increment after finishing
    completed: Arc<AtomicUsize>,
    num_threads: usize,
}

unsafe impl Send for PersistentThreadPool {}
unsafe impl Sync for PersistentThreadPool {}

impl PersistentThreadPool {
    /// Create a new persistent thread pool
    pub fn new(num_threads: usize) -> Self {
        let current_op = Arc::new(AtomicUsize::new(0));
        let work_fn: Arc<
            std::sync::RwLock<Option<Arc<dyn Fn(usize, usize) + Send + Sync>>>,
        > = Arc::new(std::sync::RwLock::new(None));
        let num_items = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let current_op = current_op.clone();
                let work_fn = work_fn.clone();
                let num_items = num_items.clone();
                let completed = completed.clone();

                thread::spawn(move || {
                    let mut last_op = 0;

                    loop {
                        // Spin-wait for new operation (sub-microsecond, matches llama.cpp)
                        let op = current_op.load(Ordering::Acquire);
                        if op == usize::MAX {
                            // Shutdown signal
                            break;
                        }

                        if op == last_op {
                            // No new work yet - spin
                            std::hint::spin_loop();
                            continue;
                        }

                        // New operation - get work parameters
                        last_op = op;
                        let n = num_items.load(Ordering::Acquire);
                        let nth = num_threads;

                        // Get function reference (Arc cloned) - use read lock
                        let func_opt = work_fn.read().unwrap().clone();
                        if let Some(func) = func_opt {
                            // Process rows assigned to this thread
                            // Round-robin: thread i handles rows i, i+nth, i+2*nth, ...
                            let mut row = thread_id;
                            while row < n {
                                func(row, row + 1);
                                row += nth;
                            }
                        }

                        // Signal completion
                        completed.fetch_add(1, Ordering::Release);
                    }
                })
            })
            .collect();

        PersistentThreadPool {
            handles,
            current_op,
            work_fn,
            num_items,
            completed,
            num_threads,
        }
    }

    /// Execute work on all threads
    /// Main thread sets work then signals workers via atomic increment
    pub fn execute<F>(&self, num_items: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        if num_items == 0 {
            return;
        }

        // Reset completion counter
        self.completed.store(0, Ordering::Release);

        // Set work parameters
        self.num_items.store(num_items, Ordering::Release);

        // Store function in Arc for safe lifetime - use write lock
        *self.work_fn.write().unwrap() = Some(Arc::new(f));

        // Signal workers by incrementing operation counter
        self.current_op.fetch_add(1, Ordering::Release);

        // Main thread waits for all workers to complete
        while self.completed.load(Ordering::Acquire) < self.num_threads {
            std::hint::spin_loop();
        }
    }

    /// Get number of threads
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Execute GEMV-style work where each thread processes rows
    /// Uses work stealing with atomic counter
    pub fn gemv_rows_stealing<F>(
        &self,
        num_rows: usize,
        chunk_size: usize,
        f: F,
    ) where
        F: Fn(usize, usize) + Send + Sync + Clone + 'static,
    {
        self.execute(num_rows, f);
    }

    /// Execute GEMV with fixed chunk assignment (cache-friendly)
    /// Each thread processes a contiguous chunk of rows
    pub fn gemv_chunks<F>(&self, num_rows: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync + Clone + 'static,
    {
        self.execute(num_rows, f);
    }
}

impl Drop for PersistentThreadPool {
    fn drop(&mut self) {
        // Signal shutdown
        self.current_op.store(usize::MAX, Ordering::Release);
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

// Global thread pool instance
static GLOBAL_POOL: std::sync::OnceLock<PersistentThreadPool> =
    std::sync::OnceLock::new();

/// Get or create the global thread pool
pub fn global_pool() -> &'static PersistentThreadPool {
    GLOBAL_POOL.get_or_init(|| PersistentThreadPool::new(default_n_threads()))
}

/// Execute GEMV with work stealing (better load balancing)
/// Uses std::thread::scope for parallel execution (no Rayon)
pub fn gemv_work_stealing<F>(num_rows: usize, num_threads: usize, f: F)
where
    F: Fn(usize) + Send + Sync,
{
    if num_rows == 0 {
        return;
    }

    // Use std::thread::scope instead of Rayon
    std::thread::scope(|s| {
        let counter =
            std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let f_ref = &f;

        for _ in 0..num_threads {
            let counter = counter.clone();
            s.spawn(move || {
                loop {
                    let row = counter
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if row >= num_rows {
                        break;
                    }
                    f_ref(row);
                }
            });
        }
    });
}

/// Wrapper for mutable slice pointer that is Send + Sync
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(*mut T);

unsafe impl<T: Send> Send for SyncPtr<T> {}
unsafe impl<T: Send> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    pub fn new(ptr: *mut T) -> Self {
        SyncPtr(ptr)
    }

    pub fn get(&self) -> *mut T {
        self.0
    }
}

/// Barrier-synchronized batch GEMV execution
/// Executes multiple GEMV operations with a single thread dispatch
/// Threads wait at barrier between operations, eliminating dispatch overhead
pub fn gemv_batch_barrier<F>(
    ops: &[GEMVOp],
    num_threads: usize,
    barrier: &SpinBarrier,
) where
    F: Fn(usize, usize) + Send + Sync,
{
    if ops.is_empty() || num_threads == 0 {
        return;
    }

    std::thread::scope(|s| {
        for thread_id in 0..num_threads {
            let barrier = barrier;
            s.spawn(move || {
                for op in ops {
                    // Process assigned rows for this operation
                    let mut row = thread_id;
                    while row < op.num_rows {
                        (op.compute)(row);
                        row += num_threads;
                    }
                    // Wait at barrier for all threads to complete this operation
                    barrier.wait();
                }
            });
        }
    });
}

/// Single GEMV operation for batch execution
pub struct GEMVOp {
    pub num_rows: usize,
    pub compute: fn(usize),
}

/// Barrier-synchronized layer execution
/// Processes an entire layer with minimal thread dispatch overhead
pub fn execute_layer_with_barrier<F>(
    num_threads: usize,
    num_ops: usize,
    op_fn: F,
) where
    F: Fn(usize, usize) + Send + Sync,
{
    if num_threads == 0 || num_ops == 0 {
        return;
    }

    let barrier = SpinBarrier::new(num_threads);

    std::thread::scope(|s| {
        for thread_id in 0..num_threads {
            let op_fn = &op_fn;
            let barrier = &barrier;
            s.spawn(move || {
                for op_idx in 0..num_ops {
                    // Execute operation for this thread
                    op_fn(thread_id, op_idx);
                    // Wait at barrier
                    barrier.wait();
                }
            });
        }
    });
}
