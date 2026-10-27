// SPDX-License-Identifier: AGPL-3.0-or-later
// CPU execution model matching llama.cpp

use once_cell::sync::Lazy;
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

/// Task function type matching llama.cpp's approach
/// Parameters: (counter_ptr, chunk_size, n_items, user_data)
pub type TaskFn = unsafe fn(*const AtomicUsize, usize, usize, usize);

/// Task for parallel execution
#[derive(Clone, Copy)]
struct Task {
    func: TaskFn,
    counter: *const AtomicUsize,
    n_items: usize,
    chunk_size: usize,
    user_data: usize, // User-provided data pointer
}

unsafe impl Send for Task {}
unsafe impl Sync for Task {}

/// Persistent thread pool matching llama.cpp's execution model
/// - Threads are created once and persist
/// - All threads process the SAME task with work stealing
/// - Barrier synchronization between tasks
pub struct CpuThreadPool {
    n_threads: usize,
    _barrier: Arc<SpinBarrier>,
    counter: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    handles: Vec<JoinHandle<()>>,
    task: Arc<Mutex<Option<Task>>>, // Current task
    task_state: Arc<(Mutex<TaskState>, Condvar)>, // State machine for synchronization
    _ready_count: Arc<AtomicUsize>, // Track how many threads are ready
}

/// Task state machine
#[derive(Clone, Copy, PartialEq)]
enum TaskState {
    NoTask,    // No task available, workers wait
    TaskReady, // Task is ready for execution
    TaskDone,  // Task has been completed
}

impl CpuThreadPool {
    /// Create a new persistent thread pool
    pub fn new(n_threads: usize) -> Self {
        let barrier = Arc::new(SpinBarrier::new(n_threads));
        let counter = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let task: Arc<Mutex<Option<Task>>> = Arc::new(Mutex::new(None));
        let task_state: Arc<(Mutex<TaskState>, Condvar)> =
            Arc::new((Mutex::new(TaskState::NoTask), Condvar::new()));
        let ready_count = Arc::new(AtomicUsize::new(0));

        let handles: Vec<JoinHandle<()>> = (0..n_threads)
            .map(|_| {
                let barrier = barrier.clone();
                let _counter = counter.clone();
                let stop = stop.clone();
                let task = task.clone();
                let task_state = task_state.clone();
                let ready_count = ready_count.clone();

                thread::spawn(move || {
                    // Signal that this thread is ready
                    ready_count.fetch_add(1, Ordering::SeqCst);

                    loop {
                        // Wait for task to be ready
                        // We accept either TaskReady or TaskDone (if we're late)
                        {
                            let (lock, cvar) = &*task_state;
                            let mut guard = lock.lock().unwrap();
                            // Wait until there's a new task (TaskReady means new task available)
                            while *guard != TaskState::TaskReady
                                && !stop.load(Ordering::Relaxed)
                            {
                                guard = cvar.wait(guard).unwrap();
                            }
                            if stop.load(Ordering::Relaxed) {
                                return;
                            }
                        }

                        // Get task copy while holding the task lock
                        let task_copy = {
                            let guard = task.lock().unwrap();
                            guard.as_ref().copied()
                        };

                        if let Some(t) = task_copy {
                            // Execute task with work stealing
                            unsafe {
                                (t.func)(
                                    t.counter,
                                    t.chunk_size,
                                    t.n_items,
                                    t.user_data,
                                );
                            }
                        }

                        // Barrier: wait for all threads to finish task execution
                        barrier.wait();

                        // After barrier: all threads have finished the task
                        // Only one thread signals completion
                        {
                            let (lock, cvar) = &*task_state;
                            let mut guard = lock.lock().unwrap();
                            // Transition from TaskReady to TaskDone
                            if *guard == TaskState::TaskReady {
                                *guard = TaskState::TaskDone;
                                cvar.notify_all();
                            }
                        }

                        // Wait for main thread to acknowledge completion
                        // We wait until state is no longer TaskDone
                        {
                            let (lock, cvar) = &*task_state;
                            let mut guard = lock.lock().unwrap();
                            while *guard == TaskState::TaskDone
                                && !stop.load(Ordering::Relaxed)
                            {
                                guard = cvar.wait(guard).unwrap();
                            }
                        }
                        // At this point, state could be NoTask or TaskReady (next task)
                        // Loop back and check for next task
                    }
                })
            })
            .collect();

        // Wait for all threads to be ready
        while ready_count.load(Ordering::SeqCst) < n_threads {
            std::hint::spin_loop();
        }

        Self {
            n_threads,
            _barrier: barrier,
            counter,
            stop,
            handles,
            task,
            task_state,
            _ready_count: ready_count,
        }
    }

    /// Execute parallel work with work stealing using raw function pointer
    /// This matches llama.cpp's approach and avoids 'static closure requirement
    pub fn execute(
        &self,
        func: TaskFn,
        n_items: usize,
        chunk_size: usize,
        user_data: usize,
    ) {
        if n_items == 0 {
            return;
        }

        // Reset counter with SeqCst ordering to ensure all threads see it
        self.counter.store(0, Ordering::SeqCst);

        let task = Task {
            func,
            counter: Arc::as_ptr(&self.counter),
            n_items,
            chunk_size,
            user_data,
        };

        // Store task
        *self.task.lock().unwrap() = Some(task);

        // Ensure task is visible before signaling
        std::sync::atomic::fence(Ordering::Release);

        // Signal task ready
        {
            let (lock, cvar) = &*self.task_state;
            *lock.lock().unwrap() = TaskState::TaskReady;
            cvar.notify_all();
        }

        // Wait for completion (state becomes TaskDone)
        {
            let (lock, cvar) = &*self.task_state;
            let mut guard = lock.lock().unwrap();
            while *guard == TaskState::TaskReady {
                guard = cvar.wait(guard).unwrap();
            }
        }

        // Reset to NoTask so workers can exit their wait loop
        // Workers are waiting for state != TaskDone
        {
            let (lock, cvar) = &*self.task_state;
            *lock.lock().unwrap() = TaskState::NoTask;
            cvar.notify_all();
        }

        // Clear task
        *self.task.lock().unwrap() = None;
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
            let (lock, cvar) = &*self.task_state;
            *lock.lock().unwrap() = TaskState::NoTask;
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

/// Global thread pool for CPU inference
/// Created lazily on first use and reused across all operations
pub static GLOBAL_THREAD_POOL: Lazy<CpuThreadPool> =
    Lazy::new(|| CpuThreadPool::new(default_n_threads()));

/// Execute work using the global persistent thread pool
/// Uses raw function pointer to avoid 'static closure requirement
pub fn parallel_for_raw(
    func: TaskFn,
    n_items: usize,
    chunk_size: usize,
    user_data: usize,
) {
    if n_items == 0 {
        return;
    }
    if n_items <= chunk_size {
        // Sequential execution for small workloads
        unsafe {
            // Create a local counter for sequential execution
            let counter = AtomicUsize::new(0);
            (func)(&counter, chunk_size, n_items, user_data);
        }
        return;
    }
    GLOBAL_THREAD_POOL.execute(func, n_items, chunk_size, user_data);
}

/// Generic work stealing loop used by task functions
/// Call this inside task functions to implement work stealing
#[inline]
pub unsafe fn work_stealing_loop<F>(
    counter: *const AtomicUsize,
    chunk_size: usize,
    n_items: usize,
    mut f: F,
) where
    F: FnMut(usize, usize),
{
    if counter.is_null() {
        // Sequential execution
        f(0, n_items);
        return;
    }

    let counter = &*counter;
    loop {
        let chunk_start = counter.fetch_add(chunk_size, Ordering::Relaxed);
        if chunk_start >= n_items {
            break;
        }
        let chunk_end = (chunk_start + chunk_size).min(n_items);
        f(chunk_start, chunk_end);
    }
}

/// Execute work in parallel using work stealing
/// This matches llama.cpp's execution model with dynamic load balancing
pub fn parallel_work_stealing<F>(n_items: usize, chunk_size: usize, f: &F)
where
    F: Fn(usize, usize) + Send + Sync,
{
    if n_items == 0 {
        return;
    }

    // For small workloads, just use sequential
    if n_items <= chunk_size {
        f(0, n_items);
        return;
    }

    let n_chunks = (n_items + chunk_size - 1) / chunk_size;
    let counter = Arc::new(WorkStealingCounter::new(0));

    thread::scope(|s| {
        let n_threads = default_n_threads().min(n_chunks);
        for _ in 0..n_threads {
            let counter = counter.clone();
            s.spawn(move || {
                loop {
                    let chunk_idx = counter.fetch_add();
                    if chunk_idx >= n_chunks {
                        break;
                    }
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(n_items);
                    f(start, end);
                }
            });
        }
    });
}

/// Execute element-wise operation in parallel with work stealing
/// Optimized for element-wise operations over slices
pub fn parallel_for_each<F>(data: &mut [f32], f: F)
where
    F: Fn(usize, &mut f32) + Send + Sync,
{
    let len = data.len();
    if len == 0 {
        return;
    }

    // For small arrays, sequential is faster (avoid thread overhead)
    if len < 1024 {
        for (i, elem) in data.iter_mut().enumerate() {
            f(i, elem);
        }
        return;
    }

    let chunk_size = 256;
    let n_chunks = (len + chunk_size - 1) / chunk_size;
    let counter = Arc::new(AtomicUsize::new(0));
    let n_threads = default_n_threads().min(n_chunks);
    let data_addr = data.as_mut_ptr() as usize;

    thread::scope(|s| {
        for _ in 0..n_threads {
            let counter = counter.clone();
            let f = &f;
            s.spawn(move || {
                let data_ptr = data_addr as *mut f32;
                loop {
                    let chunk_idx = counter.fetch_add(1, Ordering::Relaxed);
                    if chunk_idx >= n_chunks {
                        break;
                    }
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(len);
                    for i in start..end {
                        unsafe {
                            let elem = &mut *data_ptr.add(i);
                            f(i, elem);
                        }
                    }
                }
            });
        }
    });
}

/// Execute element-wise operation over two slices in parallel
pub fn parallel_zip<F>(lhs: &[f32], rhs: &[f32], out: &mut [f32], f: F)
where
    F: Fn(f32, f32) -> f32 + Send + Sync,
{
    let len = out.len();
    if len == 0 {
        return;
    }

    // For small arrays, sequential is faster
    if len < 1024 {
        for i in 0..len {
            out[i] = f(lhs[i], rhs[i]);
        }
        return;
    }

    let chunk_size = 256;
    let n_chunks = (len + chunk_size - 1) / chunk_size;
    let counter = Arc::new(AtomicUsize::new(0));
    let n_threads = default_n_threads().min(n_chunks);
    let lhs_addr = lhs.as_ptr() as usize;
    let rhs_addr = rhs.as_ptr() as usize;
    let out_addr = out.as_mut_ptr() as usize;

    thread::scope(|s| {
        for _ in 0..n_threads {
            let counter = counter.clone();
            let f = &f;
            s.spawn(move || {
                let lhs_ptr = lhs_addr as *const f32;
                let rhs_ptr = rhs_addr as *const f32;
                let out_ptr = out_addr as *mut f32;
                loop {
                    let chunk_idx = counter.fetch_add(1, Ordering::Relaxed);
                    if chunk_idx >= n_chunks {
                        break;
                    }
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(len);
                    for i in start..end {
                        unsafe {
                            *out_ptr.add(i) =
                                f(*lhs_ptr.add(i), *rhs_ptr.add(i));
                        }
                    }
                }
            });
        }
    });
}
