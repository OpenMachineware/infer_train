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

/// Parallel chunks with fixed distribution
/// Uses std::thread::scope for low-overhead execution
/// Falls back to single-threaded for small workloads to avoid spawn overhead
pub fn parallel_chunks<F>(n_items: usize, n_threads: usize, f: F)
where
    F: Fn(usize, usize) + Send + Sync + Clone,
{
    if n_items == 0 {
        return;
    }

    // Single-threaded for small workloads to avoid thread spawn overhead
    // Threshold tuned for decode: GEMV with m < 2048 is memory-bound, threads don't help
    if n_items < 2048 || n_threads == 1 {
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

/// Thread command
enum Command {
    Work(Box<dyn FnOnce() + Send + 'static>),
    Shutdown,
}

/// Persistent thread pool with barrier synchronization
///
/// Unlike std::thread::scope or Rayon, threads persist across GEMV calls,
/// eliminating thread creation/spawn overhead.
pub struct PersistentThreadPool {
    handles: Vec<JoinHandle<()>>,
    sender: crossbeam::channel::Sender<Command>,
    barrier: Arc<SpinBarrier>,
    num_threads: usize,
}

impl PersistentThreadPool {
    /// Create a new persistent thread pool
    pub fn new(num_threads: usize) -> Self {
        let (sender, receiver) = crossbeam::channel::bounded(num_threads * 2);
        // Barrier for num_threads workers + 1 main thread participant
        let barrier = Arc::new(SpinBarrier::new(num_threads + 1));
        let receiver = Arc::new(receiver);

        let handles: Vec<_> = (0..num_threads)
            .map(|_thread_id| {
                let receiver = receiver.clone();
                let barrier = barrier.clone();

                thread::spawn(move || {
                    loop {
                        // Wait for work
                        match receiver.recv() {
                            Ok(Command::Work(func)) => {
                                func();
                                // Wait at barrier for all threads to complete
                                barrier.wait();
                            }
                            Ok(Command::Shutdown) | Err(_) => break,
                        }
                    }
                })
            })
            .collect();

        PersistentThreadPool { handles, sender, barrier, num_threads }
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
        if num_rows == 0 {
            return;
        }

        let counter = Arc::new(AtomicUsize::new(0));
        let barrier = self.barrier.clone();

        for _ in 0..self.num_threads {
            let f_clone = f.clone();
            let counter = counter.clone();
            let _barrier = barrier.clone();

            self.sender
                .send(Command::Work(Box::new(move || {
                    loop {
                        let start =
                            counter.fetch_add(chunk_size, Ordering::Relaxed);
                        if start >= num_rows {
                            break;
                        }
                        let end = (start + chunk_size).min(num_rows);
                        f_clone(start, end);
                    }
                })))
                .expect("Failed to send work");
        }

        // Wait for barrier (main thread participates too)
        barrier.wait();
    }

    /// Get number of threads
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Get the barrier for external synchronization
    pub fn barrier(&self) -> Arc<SpinBarrier> {
        self.barrier.clone()
    }

    /// Execute a graph with all threads
    /// Each thread calls task_fn with (thread_id, num_threads, user_data, graph_id)
    pub fn execute_graph<F>(
        &self,
        task_fn: F,
        user_data: usize,
        graph_id: usize,
    ) where
        F: Fn(usize, usize, usize, usize) + Send + Sync + Clone + 'static,
    {
        let nth = self.num_threads;

        for ith in 0..self.num_threads {
            let task_fn = task_fn.clone();

            self.sender
                .send(Command::Work(Box::new(move || {
                    task_fn(ith, nth, user_data, graph_id);
                })))
                .expect("Failed to send work");
        }

        // Wait for barrier (main thread participates too)
        self.barrier.wait();
    }

    /// Execute GEMV with fixed chunk assignment (cache-friendly)
    /// Each thread processes a contiguous chunk of rows
    /// NO work stealing - each thread has fixed assignment
    pub fn gemv_chunks<F>(&self, num_rows: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync + Clone + 'static,
    {
        if num_rows == 0 {
            return;
        }

        let chunk_size = (num_rows + self.num_threads - 1) / self.num_threads;

        for ith in 0..self.num_threads {
            let f_clone = f.clone();
            let start = ith * chunk_size;
            let end = (start + chunk_size).min(num_rows);

            if start >= num_rows {
                break;
            }

            self.sender
                .send(Command::Work(Box::new(move || {
                    f_clone(start, end);
                })))
                .expect("Failed to send work");
        }

        // Wait for barrier (main thread participates too)
        self.barrier.wait();
    }
}

impl Drop for PersistentThreadPool {
    fn drop(&mut self) {
        for _ in 0..self.num_threads {
            let _ = self.sender.send(Command::Shutdown);
        }
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

/// Global thread pool instance (legacy name for compatibility)
pub static GLOBAL_THREAD_POOL: std::sync::LazyLock<PersistentThreadPool> =
    std::sync::LazyLock::new(|| PersistentThreadPool::new(default_n_threads()));

/// Execute GEMV using persistent threads with fixed row assignment
/// Each thread processes rows [thread_id, thread_id + n_threads, ...]
/// No work queue - each thread knows its assignment
pub fn gemv_persistent<F>(num_rows: usize, num_threads: usize, f: F)
where
    F: Fn(usize) + Send + Sync,
{
    if num_rows == 0 {
        return;
    }

    let f_ref = &f;

    // Use std::thread::scope for one-shot parallel execution
    // This is actually fine for single GEMV - overhead is minimal
    // The key is using fixed assignment, not work queue
    std::thread::scope(|s| {
        for thread_id in 0..num_threads {
            s.spawn(move || {
                // Fixed row assignment: thread i processes rows i, i+n_threads, i+2*n_threads, ...
                let mut row = thread_id;
                while row < num_rows {
                    f_ref(row);
                    row += num_threads;
                }
            });
        }
    });
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

/// Global persistent thread pool for CPU inference
/// This eliminates thread creation overhead for repeated GEMV calls
pub fn get_persistent_pool() -> &'static PersistentThreadPool {
    use once_cell::sync::Lazy;

    static POOL: Lazy<PersistentThreadPool> =
        Lazy::new(|| PersistentThreadPool::new(default_n_threads()));

    &POOL
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
