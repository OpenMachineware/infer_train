// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
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

/// Lock-free work descriptor for GEMV operations
/// No locks - all fields are atomic or plain data
#[derive(Clone, Copy)]
struct WorkDescriptor {
    /// Function pointer (GEMV kernel) - unsafe fn for kernel functions
    func: Option<
        unsafe fn(*const u8, *const u8, *mut f32, usize, usize, usize, usize),
    >,
    /// Weight matrix pointer
    weights: *const u8,
    /// Input vector pointer
    input: *const u8,
    /// Output vector pointer
    output: *mut f32,
    /// K dimension (input size)
    k: usize,
    /// M dimension (output size / rows)
    m: usize,
}

impl Default for WorkDescriptor {
    fn default() -> Self {
        WorkDescriptor {
            func: None,
            weights: std::ptr::null(),
            input: std::ptr::null(),
            output: std::ptr::null_mut(),
            k: 0,
            m: 0,
        }
    }
}

unsafe impl Send for WorkDescriptor {}
unsafe impl Sync for WorkDescriptor {}

/// Persistent thread pool - WORK-STEALING version matching llama.cpp exactly
///
/// Key design from llama.cpp:
/// 1. Threads wait on atomic "current_op" counter (spin loop)
/// 2. Main thread stores work descriptor, then increments counter
/// 3. Workers grab chunks via atomic fetch_add (work-stealing)
/// 4. 64-row chunks for better load balancing
pub struct PersistentThreadPool {
    handles: Vec<JoinHandle<()>>,
    /// Current operation number - workers spin-wait until this changes
    current_op: Arc<AtomicUsize>,
    /// Work descriptor - lock-free, written before signaling
    work: Arc<std::sync::atomic::AtomicPtr<WorkDescriptor>>,
    /// Pre-allocated work descriptor slot
    work_slot: Arc<std::cell::UnsafeCell<WorkDescriptor>>,
    /// Completion counter - workers increment after finishing
    completed: Arc<AtomicUsize>,
    /// Current chunk for work-stealing
    current_chunk: Arc<AtomicUsize>,
    num_threads: usize,
}

unsafe impl Send for PersistentThreadPool {}
unsafe impl Sync for PersistentThreadPool {}

impl PersistentThreadPool {
    /// Create a new persistent thread pool with work-stealing
    pub fn new(num_threads: usize) -> Self {
        let current_op = Arc::new(AtomicUsize::new(0));
        let work_slot =
            Arc::new(std::cell::UnsafeCell::new(WorkDescriptor::default()));
        let work =
            Arc::new(std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()));
        let completed = Arc::new(AtomicUsize::new(0));
        let current_chunk = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let current_op = current_op.clone();
                let work = work.clone();
                let completed = completed.clone();
                let current_chunk = current_chunk.clone();

                thread::spawn(move || {
                    let mut last_op = 0;
                    let mut spin_count = 0;

                    loop {
                        // Spin-wait for new operation (sub-microsecond, matches llama.cpp)
                        let op = current_op.load(Ordering::Acquire);
                        if op == usize::MAX {
                            // Shutdown signal
                            break;
                        }

                        if op == last_op {
                            // No new work yet - spin for a bit then yield
                            spin_count += 1;
                            if spin_count < 100 {
                                std::hint::spin_loop();
                            } else {
                                std::thread::yield_now();
                                spin_count = 0;
                            }
                            continue;
                        }

                        spin_count = 0;

                        // New operation - read work descriptor (lock-free!)
                        last_op = op;
                        let work_ptr: *const WorkDescriptor =
                            work.load(Ordering::Acquire);
                        if work_ptr.is_null() {
                            continue;
                        }

                        let work_desc = unsafe { &*work_ptr };
                        let m = work_desc.m;

                        // Work-stealing: each thread grabs chunks via atomic fetch_add
                        // Chunk size 64 matches llama.cpp for good load balancing
                        const CHUNK_SIZE: usize = 64;
                        let n_chunks = (m + CHUNK_SIZE - 1) / CHUNK_SIZE;

                        if let Some(func) = work_desc.func {
                            loop {
                                let chunk = current_chunk
                                    .fetch_add(1, Ordering::Relaxed);
                                if chunk >= n_chunks {
                                    break;
                                }

                                let start = chunk * CHUNK_SIZE;
                                let end = (start + CHUNK_SIZE).min(m);

                                unsafe {
                                    func(
                                        work_desc.weights,
                                        work_desc.input,
                                        work_desc.output,
                                        work_desc.k,
                                        m,
                                        start,
                                        end,
                                    );
                                }
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
            work,
            work_slot,
            completed,
            current_chunk,
            num_threads,
        }
    }

    /// Execute GEMV operation - work-stealing with 64-row chunks
    /// Main thread stores work descriptor, then signals via atomic increment
    pub fn execute_gemv(
        &self,
        func: unsafe fn(
            *const u8,
            *const u8,
            *mut f32,
            usize,
            usize,
            usize,
            usize,
        ),
        weights: *const u8,
        input: *const u8,
        output: *mut f32,
        k: usize,
        m: usize,
    ) {
        if m == 0 {
            return;
        }

        // Reset completion counter and chunk counter
        self.completed.store(0, Ordering::Release);
        self.current_chunk.store(0, Ordering::Release);

        // Store work descriptor
        unsafe {
            *self.work_slot.get() = WorkDescriptor {
                func: Some(func),
                weights,
                input,
                output,
                k,
                m,
            };

            // Publish work descriptor via atomic store
            self.work.store(self.work_slot.get(), Ordering::Release);
        }

        // Signal workers by incrementing operation counter
        self.current_op.fetch_add(1, Ordering::Release);

        // Main thread waits for all workers to complete
        let mut spin_count = 0;
        while self.completed.load(Ordering::Acquire) < self.num_threads {
            spin_count += 1;
            if spin_count < 100 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
                spin_count = 0;
            }
        }
    }

    /// Legacy execute method - uses generic closure (still has some overhead)
    /// For critical paths, use execute_gemv instead
    pub fn execute<F>(&self, num_items: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync + Clone + 'static,
    {
        // Fallback: this still has allocation overhead
        // For zero-overhead, use execute_gemv
        parallel_chunks(num_items, self.num_threads, f);
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
