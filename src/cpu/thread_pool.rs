// CPU thread pool with work-stealing GEMM
// Matches llama.cpp architecture: persistent threads, barrier sync, atomic work stealing
//
// Key design:
// - Idle waiting: cond_wait (sleep, has syscall) - correct for long waits
// - Barrier sync: spin + PAUSE/YIELD (no syscall) - correct for short waits
// - Work stealing: atomic fetch_add (no lock) - dynamic load balancing

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
    pub current_chunk: AtomicUsize,
    pub barrier: SpinBarrier,
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
        let state = Arc::new(ThreadPoolState {
            current_chunk: AtomicUsize::new(0),
            barrier: SpinBarrier::new(n_threads),
            stop: AtomicBool::new(false),
            work: Mutex::new(None),
            cond: Condvar::new(),
            work_ready: AtomicBool::new(false),
        });

        let mut handles = Vec::with_capacity(n_threads - 1);

        for ith in 1..n_threads {
            let state = Arc::clone(&state);
            let handle = thread::spawn(move || {
                loop {
                    let mut work_lock = state.work.lock().unwrap();
                    while !state.work_ready.load(Ordering::Relaxed)
                        && !state.stop.load(Ordering::Relaxed)
                    {
                        work_lock = state.cond.wait(work_lock).unwrap();
                    }

                    if state.stop.load(Ordering::Relaxed) {
                        return;
                    }

                    // Clone the Arc to the work item so all workers can execute it
                    let work =
                        work_lock.as_ref().map(|item| Arc::clone(&item.func));
                    drop(work_lock);

                    if let Some(func) = work {
                        func(ith);
                    }

                    state.barrier.wait();
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

    pub fn parallel<F>(&self, f: F)
    where
        F: Fn(usize) + Send + Sync + Clone + 'static,
    {
        self.state.current_chunk.store(self.n_threads, Ordering::Relaxed);
        self.state.barrier.reset();

        {
            let mut work_lock = self.state.work.lock().unwrap();
            *work_lock = Some(WorkItem { func: Arc::new(f.clone()) });
        }
        self.state.work_ready.store(true, Ordering::Release);
        self.state.cond.notify_all();

        f(0);

        self.state.barrier.wait();
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
    F: Fn(usize) + Send + Sync + Clone + 'static,
{
    if n_threads == 1 || n_chunks <= 1 {
        for chunk in 0..n_chunks {
            f(chunk);
        }
        return;
    }

    let pool = global_pool(n_threads);
    let f_clone = f.clone();

    pool.parallel(move |ith| {
        let mut chunk = ith;
        while chunk < n_chunks {
            f_clone(chunk);
            chunk = pool.state().current_chunk.fetch_add(1, Ordering::Relaxed);
        }
    });
}

pub fn parallel_chunks<F>(n_chunks: usize, n_threads: usize, f: F)
where
    F: Fn(usize, usize) + Send + Sync + Clone + 'static,
{
    if n_threads == 1 || n_chunks <= n_threads {
        f(0, n_chunks);
        return;
    }

    let pool = global_pool(n_threads);
    let f_clone = f.clone();

    pool.parallel(move |ith| {
        let mut chunk = ith;
        while chunk < n_chunks {
            let start = chunk * 1; // Each chunk is 1 unit
            let end = (start + 1).min(n_chunks);
            if start < n_chunks {
                f_clone(start, end);
            }
            chunk = pool.state().current_chunk.fetch_add(1, Ordering::Relaxed);
        }
    });
}

pub fn default_n_threads() -> usize {
    let n = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
    (n - 1).max(1).min(8)
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
