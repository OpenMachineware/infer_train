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

/// Persistent thread pool - LOCK-FREE version matching llama.cpp exactly
///
/// Key design from llama.cpp:
/// 1. Threads wait on atomic "current_op" counter (spin loop)
/// 2. Main thread stores work descriptor, then increments counter
/// 3. Workers read descriptor, process work, increment completion
/// 4. NO locks - pure atomic operations for sub-microsecond wake
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
    num_threads: usize,
}

unsafe impl Send for PersistentThreadPool {}
unsafe impl Sync for PersistentThreadPool {}

impl PersistentThreadPool {
    /// Create a new persistent thread pool
    pub fn new(num_threads: usize) -> Self {
        let current_op = Arc::new(AtomicUsize::new(0));
        let work_slot =
            Arc::new(std::cell::UnsafeCell::new(WorkDescriptor::default()));
        let work =
            Arc::new(std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()));
        let completed = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let current_op = current_op.clone();
                let work = work.clone();
                let completed = completed.clone();

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
                        let nth = num_threads;

                        // Calculate chunk for this thread
                        let chunk_size = (m + nth - 1) / nth;
                        let start = thread_id * chunk_size;
                        let end = (start + chunk_size).min(m);

                        // Execute GEMV for assigned rows
                        if start < m {
                            if let Some(func) = work_desc.func {
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
            num_threads,
        }
    }

    /// Execute GEMV operation - lock-free
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

        // Reset completion counter
        self.completed.store(0, Ordering::Release);

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

/// Forward pass operation types for barrier-synchronized execution
#[derive(Clone, Copy, Debug)]
pub enum ForwardOp {
    /// GEMV operation: weights, input, output, k, m
    Gemv {
        weights: *const u8,
        input: *const u8,
        output: *mut f32,
        k: usize,
        m: usize,
        dtype: u8, // 0=Q4_K, 1=Q5_K, 2=Q6_K
    },
    /// RMS Norm operation (single-thread, no parallelism needed)
    RmsNorm {
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        size: usize,
        eps: f32,
    },
    /// Barrier point - all threads must sync here
    Barrier,
    /// End of forward pass
    End,
}

unsafe impl Send for ForwardOp {}
unsafe impl Sync for ForwardOp {}

/// Work descriptor for full forward pass execution
/// Pre-computed list of all operations, threads process together
/// Work descriptor for full forward pass execution
/// Pre-computed list of all operations, threads process together
pub struct ForwardPassWork {
    /// List of operations to execute
    pub ops: Vec<ForwardOp>,
    /// Number of threads
    pub n_threads: usize,
    /// Barrier for synchronization
    barrier: Arc<SpinBarrier>,
}

impl Clone for ForwardPassWork {
    fn clone(&self) -> Self {
        ForwardPassWork {
            ops: self.ops.clone(),
            n_threads: self.n_threads,
            barrier: Arc::new(SpinBarrier::new(self.n_threads)),
        }
    }
}

impl ForwardPassWork {
    pub fn new(n_threads: usize) -> Self {
        ForwardPassWork {
            ops: Vec::with_capacity(256),
            n_threads,
            barrier: Arc::new(SpinBarrier::new(n_threads)),
        }
    }

    pub fn with_ops(ops: Vec<ForwardOp>, n_threads: usize) -> Self {
        ForwardPassWork {
            ops,
            n_threads,
            barrier: Arc::new(SpinBarrier::new(n_threads)),
        }
    }

    pub fn clear(&mut self) {
        self.ops.clear();
    }

    pub fn add_gemv(
        &mut self,
        weights: *const u8,
        input: *const u8,
        output: *mut f32,
        k: usize,
        m: usize,
        dtype: u8,
    ) {
        self.ops.push(ForwardOp::Gemv { weights, input, output, k, m, dtype });
    }

    pub fn add_barrier(&mut self) {
        self.ops.push(ForwardOp::Barrier);
    }

    pub fn add_end(&mut self) {
        self.ops.push(ForwardOp::End);
    }
}

/// Forward pass executor with barrier synchronization
/// All threads process the same operation together, like llama.cpp
pub struct ForwardPassExecutor {
    handles: Vec<JoinHandle<()>>,
    /// Current pass number - workers wait for this to change
    current_pass: Arc<AtomicUsize>,
    /// Work descriptor for the current pass
    work: Arc<std::sync::RwLock<Option<ForwardPassWork>>>,
    /// Completion counter
    completed: Arc<AtomicUsize>,
    /// Number of threads
    n_threads: usize,
}

impl ForwardPassExecutor {
    /// Create a new forward pass executor
    pub fn new(n_threads: usize) -> Self {
        let current_pass = Arc::new(AtomicUsize::new(0));
        let work = Arc::new(std::sync::RwLock::new(None));
        let completed = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..n_threads)
            .map(|thread_id| {
                let current_pass = current_pass.clone();
                let work = work.clone();
                let completed = completed.clone();
                let n_threads = n_threads;

                thread::spawn(move || {
                    let mut last_pass = 0;
                    let mut spin_count = 0;

                    loop {
                        // Spin-wait for new pass
                        let pass = current_pass.load(Ordering::Acquire);
                        if pass == usize::MAX {
                            // Shutdown signal
                            break;
                        }

                        if pass == last_pass {
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
                        last_pass = pass;

                        // Get work descriptor
                        let work_guard = work.read().unwrap();
                        let work_desc: &ForwardPassWork = match work_guard.as_ref() {
                            Some(w) => w,
                            None => continue,
                        };

                        // Process all operations
                        let ops = &work_desc.ops;
                        let n_threads_actual = work_desc.n_threads;
                        let barrier = &work_desc.barrier;

                        for op_idx in 0..ops.len() {
                            let op = ops[op_idx];

                            match op {
                                ForwardOp::Gemv {
                                    weights,
                                    input,
                                    output,
                                    k,
                                    m,
                                    dtype,
                                } => {
                                    // Parallel GEMV - each thread processes a chunk of rows
                                    let chunk_size = (m + n_threads_actual - 1) / n_threads_actual;
                                    let start = thread_id * chunk_size;
                                    let end = (start + chunk_size).min(m);

                                    if start < m {
                                        unsafe {
                                            match dtype {
                                                0 => {
                                                    // Q4_K
                                                    use crate::quant::types::BlockQ4K;
                                                    let nb = k / 256;
                                                    let weights_blocks = std::slice::from_raw_parts(
                                                        weights as *const BlockQ4K,
                                                        m * nb,
                                                    );
                                                    let input_blocks = std::slice::from_raw_parts(
                                                        input as *const crate::quant::types::BlockQ8K,
                                                        nb,
                                                    );
                                                    let output_slice =
                                                        std::slice::from_raw_parts_mut(output, m);

                                                    for row in start..end {
                                                        use crate::quant::vec_dot::arm::vec_dot_q4_k_q8_k_neon;
                                                        output_slice[row] = vec_dot_q4_k_q8_k_neon(
                                                            k,
                                                            &weights_blocks[row * nb..(row + 1) * nb],
                                                            input_blocks,
                                                        );
                                                    }
                                                }
                                                1 => {
                                                    // Q5_K
                                                    use crate::quant::types::BlockQ5K;
                                                    let nb = k / 256;
                                                    let weights_blocks = std::slice::from_raw_parts(
                                                        weights as *const BlockQ5K,
                                                        m * nb,
                                                    );
                                                    let input_blocks = std::slice::from_raw_parts(
                                                        input as *const crate::quant::types::BlockQ8K,
                                                        nb,
                                                    );
                                                    let output_slice =
                                                        std::slice::from_raw_parts_mut(output, m);

                                                    for row in start..end {
                                                        use crate::quant::vec_dot::arm::vec_dot_q5_k_q8_k_neon;
                                                        output_slice[row] = vec_dot_q5_k_q8_k_neon(
                                                            k,
                                                            &weights_blocks[row * nb..(row + 1) * nb],
                                                            input_blocks,
                                                        );
                                                    }
                                                }
                                                2 => {
                                                    // Q6_K
                                                    use crate::quant::types::BlockQ6K;
                                                    let nb = k / 256;
                                                    let weights_blocks = std::slice::from_raw_parts(
                                                        weights as *const BlockQ6K,
                                                        m * nb,
                                                    );
                                                    let input_blocks = std::slice::from_raw_parts(
                                                        input as *const crate::quant::types::BlockQ8K,
                                                        nb,
                                                    );
                                                    let output_slice =
                                                        std::slice::from_raw_parts_mut(output, m);

                                                    for row in start..end {
                                                        use crate::quant::vec_dot::arm::vec_dot_q6_k_q8_k_neon;
                                                        output_slice[row] = vec_dot_q6_k_q8_k_neon(
                                                            k,
                                                            &weights_blocks[row * nb..(row + 1) * nb],
                                                            input_blocks,
                                                        );
                                                    }
                                                }
                                                3 => {
                                                    // F16 - use F16 FMLA like matmul_f16_f32_single_thread
                                                    use crate::ops::attention::vec_dot_f16_neon;
                                                    use crate::ops::arm::neon::dup_f32_to_f16_neon;
                                                    use half::f16;

                                                    let weights_f16 = std::slice::from_raw_parts(
                                                        weights as *const u16, m * k
                                                    );
                                                    let input_f32 = std::slice::from_raw_parts(input as *const f32, k);
                                                    let output_slice = std::slice::from_raw_parts_mut(output, m);

                                                    // Convert input F32 to F16 once per thread
                                                    let mut input_f16 = vec![0u16; k];
                                                    {
                                                        let input_f16_slice = std::slice::from_raw_parts_mut(
                                                            input_f16.as_mut_ptr() as *mut f16, k
                                                        );
                                                        dup_f32_to_f16_neon(input_f32, input_f16_slice);
                                                    }

                                                    // Process rows using F16 FMLA
                                                    for row in start..end {
                                                        output_slice[row] = vec_dot_f16_neon(
                                                            &weights_f16[row * k..(row + 1) * k],
                                                            &input_f16,
                                                            k,
                                                        );
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                                ForwardOp::Barrier => {
                                    // Wait for all threads
                                    barrier.wait();
                                }
                                ForwardOp::End => {
                                    // Signal completion and exit loop
                                    break;
                                }
                                _ => {}
                            }
                        }

                        // Signal completion
                        completed.fetch_add(1, Ordering::Release);
                    }
                })
            })
            .collect();

        ForwardPassExecutor {
            handles,
            current_pass,
            work,
            completed,
            n_threads,
        }
    }

    /// Execute a forward pass
    /// All threads process the operations together
    pub fn execute(&self, mut work: ForwardPassWork) {
        // Add end marker
        work.add_end();

        // Reset completion counter
        self.completed.store(0, Ordering::Release);

        // Store work
        *self.work.write().unwrap() = Some(work);

        // Signal start by incrementing pass counter
        self.current_pass.fetch_add(1, Ordering::Release);

        // Wait for all threads to complete
        let mut spin_count = 0;
        while self.completed.load(Ordering::Acquire) < self.n_threads {
            spin_count += 1;
            if spin_count < 100 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
                spin_count = 0;
            }
        }

        // Clear work
        *self.work.write().unwrap() = None;
    }

    /// Get number of threads
    pub fn n_threads(&self) -> usize {
        self.n_threads
    }
}

impl Drop for ForwardPassExecutor {
    fn drop(&mut self) {
        // Signal shutdown
        self.current_pass.store(usize::MAX, Ordering::Release);
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

// Global forward pass executor
static GLOBAL_FORWARD_EXECUTOR: std::sync::OnceLock<ForwardPassExecutor> =
    std::sync::OnceLock::new();

/// Get or create the global forward pass executor
pub fn global_forward_executor() -> &'static ForwardPassExecutor {
    GLOBAL_FORWARD_EXECUTOR
        .get_or_init(|| ForwardPassExecutor::new(default_n_threads()))
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
