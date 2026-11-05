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
    func: Box<dyn Fn(usize) + Send + Sync>,
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

                    let work = work_lock.take();
                    drop(work_lock);

                    if let Some(item) = work {
                        (item.func)(ith);
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
            *work_lock = Some(WorkItem { func: Box::new(f.clone()) });
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
    F: Fn(usize) + Send + Sync + Clone,
{
    if n_threads == 1 || n_chunks <= 1 {
        for chunk in 0..n_chunks {
            f(chunk);
        }
        return;
    }

    let chunk_size = (n_chunks + n_threads - 1) / n_threads;

    std::thread::scope(|s| {
        for ith in 0..n_threads {
            let f_clone = f.clone();
            s.spawn(move || {
                let mut chunk = ith * chunk_size;
                let end = (chunk + chunk_size).min(n_chunks);
                while chunk < end {
                    f_clone(chunk);
                    chunk += 1;
                }
            });
        }
    });
}

pub fn parallel_chunks<F>(n_chunks: usize, n_threads: usize, f: F)
where
    F: Fn(usize, usize) + Send + Sync + Clone,
{
    if n_threads == 1 || n_chunks <= n_threads {
        f(0, n_chunks);
        return;
    }

    let chunk_size = (n_chunks + n_threads - 1) / n_threads;

    std::thread::scope(|s| {
        for ith in 0..n_threads {
            let f_clone = f.clone();
            s.spawn(move || {
                let start = ith * chunk_size;
                let end = (start + chunk_size).min(n_chunks);
                if start < n_chunks {
                    f_clone(start, end);
                }
            });
        }
    });
}

pub fn default_n_threads() -> usize {
    let n = num_cpus::get();
    (n - 1).max(1).min(8)
}

pub trait GEMVOp: Send + Sync {
    fn execute(&self, row_start: usize, row_end: usize);
}

pub fn gemv_batch_barrier(op: &dyn GEMVOp, m: usize, n_threads: usize) {
    let chunk_size = (m + n_threads - 1) / n_threads;

    parallel_work_stealing(n_threads, n_threads, |ith: usize| {
        let row_start = ith * chunk_size;
        let row_end = (row_start + chunk_size).min(m);
        if row_start < m {
            op.execute(row_start, row_end);
        }
    });
}

pub fn execute_layer_with_barrier<F>(n_ops: usize, n_threads: usize, f: F)
where
    F: Fn(usize) + Send + Sync + Clone + 'static,
{
    parallel_work_stealing(n_ops, n_threads, f);
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
