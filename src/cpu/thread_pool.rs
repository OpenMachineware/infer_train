// CPU thread pool with work-stealing GEMM
// Matches llama.cpp architecture: persistent threads, NO barrier inside operations
//
// Key design:
// - Idle waiting: cond_wait (sleep, has syscall) - correct for long waits
// - Completion detection: atomic counter + spin-wait - NO barrier inside operations
// - Work stealing: atomic fetch_add (no lock) - dynamic load balancing
//
// Critical difference from old design:
// - OLD: barrier.wait() inside every parallel() call -> overhead
// - NEW: completion counter, main thread spins until done -> no barrier overhead

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn cpu_relax() {
    unsafe { std::arch::asm!("yield", options(nostack, preserves_flags)) }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn cpu_relax() {
    unsafe { std::arch::asm!("pause", options(nostack, preserves_flags)) }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[inline(always)]
fn cpu_relax() {
    std::hint::spin_loop()
}

/// Barrier using spin-wait (no system calls)
/// Used for multi-phase algorithms where phases must be strictly separated
pub struct SpinBarrier {
    n_threads: usize,
    counter: AtomicUsize,
    passed: AtomicUsize,
}

impl SpinBarrier {
    pub fn new(n_threads: usize) -> Self {
        Self {
            n_threads,
            counter: AtomicUsize::new(0),
            passed: AtomicUsize::new(0),
        }
    }

    pub fn wait(&self) {
        if self.n_threads == 1 {
            return;
        }

        let n_passed = self.passed.load(Ordering::Relaxed);
        let n = self.counter.fetch_add(1, Ordering::SeqCst);

        if n == self.n_threads - 1 {
            self.counter.store(0, Ordering::Relaxed);
            self.passed.fetch_add(1, Ordering::SeqCst);
        } else {
            while self.passed.load(Ordering::Relaxed) == n_passed {
                cpu_relax();
            }
        }
    }

    pub fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
        self.passed.store(0, Ordering::Relaxed);
    }
}

struct WorkItem {
    func: Arc<dyn Fn(usize) + Send + Sync>,
}

pub struct ThreadPoolState {
    /// Work-stealing counter: next chunk to process
    pub current_chunk: AtomicUsize,
    /// Completion counter: how many threads have finished
    n_completed: AtomicUsize,
    /// Number of worker threads (excluding main thread)
    n_workers: usize,
    pub stop: AtomicBool,
    work: Mutex<Option<WorkItem>>,
    cond: Condvar,
    work_ready: AtomicBool,
}

pub struct ThreadPool {
    handles: Vec<JoinHandle<()>>,
    state: Arc<ThreadPoolState>,
    n_threads: usize,
}

impl ThreadPool {
    pub fn new(n_threads: usize) -> Self {
        let n_workers = if n_threads > 1 { n_threads - 1 } else { 0 };
        let state = Arc::new(ThreadPoolState {
            current_chunk: AtomicUsize::new(0),
            n_completed: AtomicUsize::new(0),
            n_workers,
            stop: AtomicBool::new(false),
            work: Mutex::new(None),
            cond: Condvar::new(),
            work_ready: AtomicBool::new(false),
        });

        let mut handles = Vec::with_capacity(n_workers);

        for ith in 1..n_threads {
            let state = Arc::clone(&state);
            let handle = thread::spawn(move || {
                loop {
                    // Wait for work
                    let mut work_lock = state.work.lock().unwrap();
                    while !state.work_ready.load(Ordering::Relaxed)
                        && !state.stop.load(Ordering::Relaxed)
                    {
                        work_lock = state.cond.wait(work_lock).unwrap();
                    }

                    if state.stop.load(Ordering::Relaxed) {
                        return;
                    }

                    // Clone the Arc to the work item
                    let work =
                        work_lock.as_ref().map(|item| Arc::clone(&item.func));
                    drop(work_lock);

                    if let Some(func) = work {
                        func(ith);
                    }

                    // Signal completion via atomic counter (NO barrier)
                    state.n_completed.fetch_add(1, Ordering::Release);
                }
            });
            handles.push(handle);
        }

        Self { handles, state, n_threads }
    }

    pub fn state(&self) -> &ThreadPoolState {
        &self.state
    }

    pub fn n_threads(&self) -> usize {
        self.n_threads
    }

    /// Execute parallel work with work-stealing
    /// NO barrier inside - uses completion counter for synchronization
    pub fn parallel<F>(&self, f: F)
    where
        F: Fn(usize) + Send + Sync + Clone + 'static,
    {
        // Reset state
        self.state.current_chunk.store(self.n_threads, Ordering::Relaxed);
        self.state.n_completed.store(0, Ordering::Relaxed);

        {
            let mut work_lock = self.state.work.lock().unwrap();
            *work_lock = Some(WorkItem { func: Arc::new(f.clone()) });
        }
        self.state.work_ready.store(true, Ordering::Release);
        self.state.cond.notify_all();

        // Main thread (ith=0) also executes work
        f(0);

        // Wait for completion via spin-wait (no barrier, no syscall)
        let n_workers = self.state.n_workers;
        if n_workers > 0 {
            while self.state.n_completed.load(Ordering::Acquire) < n_workers {
                cpu_relax();
            }
        }

        self.state.work_ready.store(false, Ordering::Relaxed);
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.state.stop.store(true, Ordering::Relaxed);
        self.state.work_ready.store(true, Ordering::Relaxed);
        self.state.cond.notify_all();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

use std::sync::OnceLock;

static THREAD_POOL: OnceLock<ThreadPool> = OnceLock::new();

pub fn global_pool(n_threads: usize) -> &'static ThreadPool {
    THREAD_POOL.get_or_init(|| ThreadPool::new(n_threads))
}

pub fn parallel_work_stealing<F>(n_chunks: usize, n_threads: usize, f: F)
where
    F: Fn(usize) + Send + Sync + Clone,
{
    if n_threads == 1 || n_chunks <= 1 {
        for chunk in 0..n_chunks {
            f(chunk);
        }
        return;
    }

    // Use std::thread::scope for lightweight parallelism without barrier deadlock
    // Each thread starts with its ID and uses atomic work stealing
    let current_chunk = AtomicUsize::new(n_threads);

    std::thread::scope(|s| {
        for ith in 0..n_threads {
            let f_clone = f.clone();
            let current_chunk = &current_chunk;

            s.spawn(move || {
                let mut chunk = ith;
                while chunk < n_chunks {
                    f_clone(chunk);
                    chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
}

/// Parallel execution with dynamic chunking and work-stealing
/// Matches llama.cpp's strategy:
/// - chunk_size: 16 for GEMM, 64 for GEMV
/// - work-stealing via atomic fetch_add
/// - redistribute chunks if nchunk0 * nchunk1 < nth * 4
pub fn parallel_chunks<F>(n_total: usize, n_threads: usize, f: F)
where
    F: Fn(usize, usize) + Send + Sync + Clone,
{
    if n_threads == 1 || n_total <= n_threads {
        f(0, n_total);
        return;
    }

    // Calculate chunk size matching llama.cpp's strategy
    // GEMV (single column): chunk_size = 64
    // GEMM (multiple columns): chunk_size = 16
    // Detect by checking n_total relative to n_threads:
    // - If n_total is large (> n_threads * 16), use 16 (GEMM)
    // - If n_total is medium, use larger chunks (GEMV)
    let chunk_size = if n_total >= n_threads * 16 { 16 } else { 64 };
    let n_chunks = (n_total + chunk_size - 1) / chunk_size;

    // Work stealing: counter starts at n_threads so threads begin with their ID
    // Thread 0 processes chunk 0, thread 1 processes chunk 1, etc.
    // Then all threads compete for remaining chunks via fetch_add
    let current_chunk = AtomicUsize::new(n_threads);

    std::thread::scope(|s| {
        for ith in 0..n_threads {
            let f_clone = f.clone();
            let current_chunk = &current_chunk;

            s.spawn(move || {
                // Thread starts with its assigned chunk
                let mut chunk_idx = ith;

                while chunk_idx < n_chunks {
                    // Calculate row range for this chunk
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(n_total);

                    if start < n_total {
                        f_clone(start, end);
                    }

                    // Get next chunk via work-stealing
                    chunk_idx = current_chunk.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
}

/// Parallel execution with explicit chunk size (for advanced use cases)
/// Allows caller to specify exact chunk size instead of auto-detection
pub fn parallel_chunks_sized<F>(
    n_total: usize,
    chunk_size: usize,
    n_threads: usize,
    f: F,
) where
    F: Fn(usize, usize) + Send + Sync + Clone,
{
    if n_threads == 1 || n_total <= n_threads {
        f(0, n_total);
        return;
    }

    let n_chunks = (n_total + chunk_size - 1) / chunk_size;
    let current_chunk = AtomicUsize::new(n_threads);

    std::thread::scope(|s| {
        for ith in 0..n_threads {
            let f_clone = f.clone();
            let current_chunk = &current_chunk;

            s.spawn(move || {
                let mut chunk_idx = ith;
                while chunk_idx < n_chunks {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(n_total);
                    if start < n_total {
                        f_clone(start, end);
                    }
                    chunk_idx = current_chunk.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });
}

/// Parallel execution using persistent ThreadPool (for prefill GEMM)
/// Uses completion counter instead of std::thread::scope - 95x lower overhead
///
/// IMPORTANT: This function requires `F` to be `'static + Clone` because the closure
/// is stored in an Arc<dyn Fn>. Use SyncPtr to pass data pointers.
pub fn parallel_chunks_pool<F>(
    n_total: usize,
    chunk_size: usize,
    n_threads: usize,
    f: F,
) where
    F: Fn(usize, usize) + Send + Sync + Clone + 'static,
{
    if n_threads == 1 || n_total <= n_threads {
        f(0, n_total);
        return;
    }

    let pool = global_pool(n_threads);
    let n_chunks = (n_total + chunk_size - 1) / chunk_size;

    // Reset work-stealing counter so threads start with their ID
    pool.state().current_chunk.store(n_threads, Ordering::Relaxed);

    pool.parallel(move |ith| {
        let mut chunk_idx = ith;
        while chunk_idx < n_chunks {
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(n_total);
            if start < n_total {
                f(start, end);
            }
            // Get next chunk via work-stealing
            chunk_idx =
                pool.state().current_chunk.fetch_add(1, Ordering::Relaxed);
        }
    });
}

pub fn default_n_threads() -> usize {
    let n = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
    (n - 1).max(1).min(8)
}

/// Thread configuration for inference
/// Matches llama.cpp's -t (decode) and -tb (prefill/batch) parameters
#[derive(Debug, Clone, Copy)]
pub struct ThreadConfig {
    /// Number of threads for decode (single token generation)
    pub n_threads_decode: usize,
    /// Number of threads for prefill (batch processing)
    pub n_threads_prefill: usize,
}

impl Default for ThreadConfig {
    fn default() -> Self {
        let default_threads = default_n_threads();
        Self {
            n_threads_decode: default_threads,
            n_threads_prefill: default_threads,
        }
    }
}

impl ThreadConfig {
    /// Create with default thread count
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom thread counts
    pub fn with_threads(decode: usize, prefill: usize) -> Self {
        Self { n_threads_decode: decode, n_threads_prefill: prefill }
    }

    /// Get thread count for the given mode
    pub fn n_threads(&self, is_decode: bool) -> usize {
        if is_decode { self.n_threads_decode } else { self.n_threads_prefill }
    }
}

pub struct SyncPtr<T> {
    ptr: *mut T,
}

unsafe impl<T: Send> Send for SyncPtr<T> {}
unsafe impl<T: Sync> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    pub fn new(ptr: *mut T) -> Self {
        Self { ptr }
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    pub fn get(&self) -> *mut T {
        self.ptr
    }

    pub unsafe fn as_ref(&self) -> &T {
        &*self.ptr
    }

    pub unsafe fn as_mut(&self) -> &mut T {
        &mut *self.ptr
    }
}

impl<T> Clone for SyncPtr<T> {
    fn clone(&self) -> Self {
        Self { ptr: self.ptr }
    }
}

impl<T> Copy for SyncPtr<T> {}
