// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// High-performance parallel execution for CPU inference
// Based on llama.cpp's thread allocation strategy

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// Get number of available CPU cores
fn num_cpus() -> usize {
    thread::available_parallelism().map(|p| p.get()).unwrap_or(1)
}

/// Shared state for thread pool
struct PoolState {
    n_chunks: AtomicUsize,
    work_fn: AtomicUsize,
    work_data: AtomicUsize,
    max_active: AtomicI32, // Maximum threads to participate (from max_threads param)
    work_generation: AtomicUsize, // Incremented for each new work batch
    work_ready: Mutex<(usize, bool)>, // (generation, shutdown)
    work_ready_cvar: Condvar,
    counter: AtomicI32,
    active: AtomicI32,
    done: Mutex<bool>,
    done_cvar: Condvar,
}

/// Persistent thread pool (llama.cpp style)
pub struct ThreadPool {
    n_threads: usize, // Total threads in pool
    threads: Vec<JoinHandle<()>>,
    state: Arc<PoolState>,
}

impl ThreadPool {
    pub fn new(n_threads: usize) -> Self {
        let n_threads =
            if n_threads == 0 { num_cpus() } else { n_threads.min(num_cpus()) };

        let state = Arc::new(PoolState {
            n_chunks: AtomicUsize::new(0),
            work_fn: AtomicUsize::new(0),
            work_data: AtomicUsize::new(0),
            max_active: AtomicI32::new(n_threads as i32), // Default to all threads
            work_generation: AtomicUsize::new(0),
            work_ready: Mutex::new((0, false)),
            work_ready_cvar: Condvar::new(),
            counter: AtomicI32::new(0),
            active: AtomicI32::new(0),
            done: Mutex::new(false),
            done_cvar: Condvar::new(),
        });

        if n_threads <= 1 {
            return Self { n_threads: 1, threads: Vec::new(), state };
        }

        // Create n_threads - 1 worker threads (main thread is the nth thread)
        let n_workers = n_threads - 1;
        let mut threads = Vec::with_capacity(n_workers);

        for tid in 1..=n_workers {
            let s = state.clone();
            let handle = thread::spawn(move || {
                let mut my_gen = 0;
                loop {
                    // Wait for new generation
                    {
                        let mut guard = s.work_ready.lock().unwrap();
                        while guard.0 == my_gen && !guard.1 {
                            guard = s.work_ready_cvar.wait(guard).unwrap();
                        }
                        if guard.1 {
                            return; // Shutdown
                        }
                        my_gen = guard.0; // Update to new generation
                    }

                    // Check if this thread should participate (tid must be < max_active)
                    let max_active =
                        s.max_active.load(Ordering::Relaxed) as usize;
                    if tid >= max_active {
                        // This thread should not participate, just signal done
                        if s.active.fetch_sub(1, Ordering::AcqRel) == 1 {
                            let mut guard = s.done.lock().unwrap();
                            *guard = true;
                            s.done_cvar.notify_all();
                        }
                        continue;
                    }

                    let n_chunks = s.n_chunks.load(Ordering::Relaxed);
                    let work_fn: fn(usize, usize, *const u8) = unsafe {
                        std::mem::transmute(s.work_fn.load(Ordering::Relaxed))
                    };
                    let work_data =
                        s.work_data.load(Ordering::Relaxed) as *const u8;

                    // Work stealing
                    loop {
                        let chunk =
                            s.counter.fetch_add(1, Ordering::Relaxed) as usize;
                        if chunk >= n_chunks {
                            break;
                        }
                        work_fn(chunk, tid, work_data);
                    }

                    // Decrement active and signal done if last thread
                    if s.active.fetch_sub(1, Ordering::AcqRel) == 1 {
                        let mut guard = s.done.lock().unwrap();
                        *guard = true;
                        s.done_cvar.notify_all();
                    }
                }
            });
            threads.push(handle);
        }

        Self { n_threads, threads, state }
    }

    /// Execute work in parallel with specified max threads
    pub unsafe fn execute_with_max(
        &self,
        n_chunks: usize,
        max_threads: usize,
        work_fn: fn(usize, usize, *const u8),
        work_data: *const u8,
    ) {
        if n_chunks == 0 || self.n_threads <= 1 {
            for chunk in 0..n_chunks {
                work_fn(chunk, 0, work_data);
            }
            return;
        }

        // Clamp max_threads to pool size
        let max_threads = max_threads.min(self.n_threads).max(1);

        // Increment generation to wake all workers
        let new_gen;
        {
            let mut guard = self.state.work_ready.lock().unwrap();
            new_gen = guard.0 + 1;
            guard.0 = new_gen;

            self.state.counter.store(0, Ordering::Relaxed);
            self.state.max_active.store(max_threads as i32, Ordering::Relaxed);
            // active = all pool threads (they will check max_active and opt out if needed)
            self.state.active.store(self.n_threads as i32, Ordering::Relaxed);
            *self.state.done.lock().unwrap() = false;
            self.state.n_chunks.store(n_chunks, Ordering::Relaxed);
            self.state.work_fn.store(work_fn as usize, Ordering::Relaxed);
            self.state.work_data.store(work_data as usize, Ordering::Relaxed);
        }
        self.state.work_ready_cvar.notify_all();

        // Main thread works too (tid = 0) if max_threads >= 1
        if max_threads >= 1 {
            loop {
                let chunk =
                    self.state.counter.fetch_add(1, Ordering::Relaxed) as usize;
                if chunk >= n_chunks {
                    break;
                }
                work_fn(chunk, 0, work_data);
            }
        }

        // Main thread done, decrement active and signal done if last
        if self.state.active.fetch_sub(1, Ordering::AcqRel) == 1 {
            let mut guard = self.state.done.lock().unwrap();
            *guard = true;
            self.state.done_cvar.notify_all();
        }

        // Wait for completion
        let mut guard = self.state.done.lock().unwrap();
        while !*guard {
            guard = self.state.done_cvar.wait(guard).unwrap();
        }
    }

    /// Execute work in parallel using all threads
    pub unsafe fn execute(
        &self,
        n_chunks: usize,
        work_fn: fn(usize, usize, *const u8),
        work_data: *const u8,
    ) {
        self.execute_with_max(n_chunks, self.n_threads, work_fn, work_data);
    }

    pub fn num_threads(&self) -> usize {
        self.n_threads
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        {
            let mut guard = self.state.work_ready.lock().unwrap();
            guard.1 = true; // Set shutdown flag
        }
        self.state.work_ready_cvar.notify_all();
        for h in self.threads.drain(..) {
            let _ = h.join();
        }
    }
}

/// Global thread pool
static GLOBAL_POOL: once_cell::sync::Lazy<ThreadPool> =
    once_cell::sync::Lazy::new(|| ThreadPool::new(0));

/// Parallel execution with max_threads support
pub unsafe fn parallel_stealing_raw(
    n_chunks: usize,
    max_threads: usize,
    work_fn: fn(usize, usize, *const u8),
    work_data: *const u8,
) {
    GLOBAL_POOL.execute_with_max(n_chunks, max_threads, work_fn, work_data);
}

/// Fixed distribution parallel (for compatibility)
pub fn parallel_fixed<F>(n_chunks: usize, n_threads: usize, f: &F)
where
    F: Fn(usize, usize, usize) + Send + Sync,
{
    if n_chunks == 0 {
        return;
    }
    let n_threads =
        if n_threads == 0 { num_cpus() } else { n_threads.min(num_cpus()) };
    if n_threads <= 1 {
        f(0, n_chunks, 0);
        return;
    }

    let chunk_size = (n_chunks + n_threads - 1) / n_threads;
    thread::scope(|s| {
        for tid in 0..n_threads {
            s.spawn(move || {
                let start = tid * chunk_size;
                let end = (start + chunk_size).min(n_chunks);
                if start < n_chunks {
                    f(start, end, tid);
                }
            });
        }
    });
}

/// Optimal thread count (llama.cpp chunking strategy)
pub fn optimal_threads(m: usize, n: usize, _k: usize) -> usize {
    let n_cpus = num_cpus();
    let mut chunk_size = 256usize;
    let mut nchunk0 = (m + chunk_size - 1) / chunk_size;
    let mut nchunk1 = (n + chunk_size - 1) / chunk_size;

    while nchunk0 * nchunk1 < n_cpus * 4 && chunk_size > 16 {
        chunk_size /= 2;
        nchunk0 = (m + chunk_size - 1) / chunk_size;
        nchunk1 = (n + chunk_size - 1) / chunk_size;
    }
    (nchunk0 * nchunk1).min(n_cpus)
}

/// Spin barrier
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

    pub fn wait(&self) {
        if self.n_threads <= 1 {
            return;
        }
        let cur_gen = self.generation.load(Ordering::Relaxed);
        let count = self.counter.fetch_add(1, Ordering::SeqCst);
        if count == (self.n_threads - 1) as i32 {
            self.counter.store(0, Ordering::Relaxed);
            self.generation.fetch_add(1, Ordering::SeqCst);
        } else {
            while self.generation.load(Ordering::Relaxed) == cur_gen {
                std::hint::spin_loop();
            }
        }
    }
}
