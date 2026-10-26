// SPDX-License-Identifier: AGPL-3.0-or-later
// CPU execution model matching llama.cpp

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// Default number of threads to use
pub fn default_n_threads() -> usize {
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4)
}

/// Spin barrier for low-latency synchronization
pub struct SpinBarrier {
    n_threads: usize,
    n_barrier: AtomicUsize,
    n_barrier_passed: AtomicUsize,
}

impl SpinBarrier {
    pub fn new(n_threads: usize) -> Self {
        Self {
            n_threads,
            n_barrier: AtomicUsize::new(0),
            n_barrier_passed: AtomicUsize::new(0),
        }
    }

    pub fn wait(&self) {
        let n_passed = self.n_barrier_passed.load(Ordering::Relaxed);
        let n_barrier = self.n_barrier.fetch_add(1, Ordering::SeqCst);

        if n_barrier == self.n_threads - 1 {
            self.n_barrier.store(0, Ordering::Relaxed);
            self.n_barrier_passed.fetch_add(1, Ordering::SeqCst);
        } else {
            #[cfg(target_arch = "aarch64")]
            while self.n_barrier_passed.load(Ordering::Relaxed) == n_passed {
                unsafe {
                    std::arch::asm!(
                        "yield",
                        options(nomem, nostack, preserves_flags)
                    );
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            while self.n_barrier_passed.load(Ordering::Relaxed) == n_passed {
                std::hint::spin_loop();
            }
        }
        std::sync::atomic::fence(Ordering::SeqCst);
    }
}

/// Work stealing counter for dynamic load balancing
pub struct WorkStealingCounter {
    current: AtomicUsize,
}

impl WorkStealingCounter {
    pub fn new(start: usize) -> Self {
        Self { current: AtomicUsize::new(start) }
    }

    pub fn fetch_add(&self) -> usize {
        self.current.fetch_add(1, Ordering::Relaxed)
    }

    pub fn reset(&self, value: usize) {
        self.current.store(value, Ordering::Relaxed);
    }
}

/// Task for GEMV with work stealing
struct GemvTask {
    n_rows: usize,
    chunk_size: usize,
    // Function pointer for the actual computation
    compute: fn(usize, usize), // row_start, row_end
}

/// Persistent thread pool matching llama.cpp's execution model
/// - Threads are created once and persist
/// - All threads process the SAME task with work stealing
/// - Barrier synchronization between tasks
pub struct CpuThreadPool {
    n_threads: usize,
    barrier: Arc<SpinBarrier>,
    counter: Arc<WorkStealingCounter>,
    stop: Arc<AtomicBool>,
    handles: Vec<JoinHandle<()>>,
    // Task state
    task_ready: Arc<(Mutex<Option<GemvTask>>, Condvar)>,
    task_done: Arc<(Mutex<bool>, Condvar)>,
}

impl CpuThreadPool {
    /// Create a new persistent thread pool
    pub fn new(n_threads: usize) -> Self {
        let barrier = Arc::new(SpinBarrier::new(n_threads));
        let counter = Arc::new(WorkStealingCounter::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let task_ready: Arc<(Mutex<Option<GemvTask>>, Condvar)> =
            Arc::new((Mutex::new(None), Condvar::new()));
        let task_done: Arc<(Mutex<bool>, Condvar)> =
            Arc::new((Mutex::new(false), Condvar::new()));

        let handles: Vec<JoinHandle<()>> = (0..n_threads)
            .map(|thread_id| {
                let barrier = barrier.clone();
                let counter = counter.clone();
                let stop = stop.clone();
                let task_ready = task_ready.clone();
                let task_done = task_done.clone();

                thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        // Wait for task
                        let (lock, cvar) = &*task_ready;
                        let task = {
                            let mut guard = lock.lock().unwrap();
                            while guard.is_none()
                                && !stop.load(Ordering::Relaxed)
                            {
                                guard = cvar.wait(guard).unwrap();
                            }
                            if stop.load(Ordering::Relaxed) {
                                return;
                            }
                            guard.take()
                        };

                        if let Some(task) = task {
                            // Work stealing: each thread grabs chunks via atomic counter
                            let n_chunks = (task.n_rows + task.chunk_size - 1)
                                / task.chunk_size;
                            loop {
                                let chunk_idx = counter.fetch_add();
                                if chunk_idx >= n_chunks {
                                    break;
                                }

                                let row_start = chunk_idx * task.chunk_size;
                                let row_end = (row_start + task.chunk_size)
                                    .min(task.n_rows);

                                (task.compute)(row_start, row_end);
                            }

                            // Signal completion
                            barrier.wait();
                            if thread_id == 0 {
                                let (lock, cvar) = &*task_done;
                                *lock.lock().unwrap() = true;
                                cvar.notify_all();
                            }
                        }
                    }
                })
            })
            .collect();

        Self {
            n_threads,
            barrier,
            counter,
            stop,
            handles,
            task_ready,
            task_done,
        }
    }

    /// Execute parallel work with work stealing
    pub fn parallel_work(
        &self,
        n_rows: usize,
        chunk_size: usize,
        compute: fn(usize, usize),
    ) {
        self.counter.reset(0);

        // Set task
        {
            let (lock, cvar) = &*self.task_ready;
            *lock.lock().unwrap() =
                Some(GemvTask { n_rows, chunk_size, compute });
            cvar.notify_all();
        }

        // Wait for completion
        {
            let (lock, cvar) = &*self.task_done;
            let mut guard = lock.lock().unwrap();
            while !*guard {
                guard = cvar.wait(guard).unwrap();
            }
            *guard = false;
        }
    }

    /// Get number of threads
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }
}

impl Drop for CpuThreadPool {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        {
            let (lock, cvar) = &*self.task_ready;
            *lock.lock().unwrap() = None;
            cvar.notify_all();
        }
        for handle in self.handles.drain(..) {
            handle.join().ok();
        }
    }
}

/// Execute work in parallel, splitting the slice into non-overlapping chunks
/// This is the safest way to do parallel writes in Rust
pub fn parallel_chunks<F, T>(data: &mut [T], n_threads: usize, f: F)
where
    F: Fn(usize, &mut [T]) + Sync + Send,
    T: Send,
{
    if data.is_empty() || n_threads <= 1 {
        f(0, data);
        return;
    }

    let len = data.len();
    let actual_threads = n_threads.min(len);
    let chunk_size = (len + actual_threads - 1) / actual_threads;

    let f_ref = &f;

    thread::scope(|s| {
        let mut remaining = data;
        let mut thread_id = 0;

        while !remaining.is_empty() {
            let split_point = chunk_size.min(remaining.len());
            let (chunk, rest) = remaining.split_at_mut(split_point);
            remaining = rest;

            let tid = thread_id;
            thread_id += 1;

            s.spawn(move || {
                f_ref(tid, chunk);
            });
        }
    });
}
