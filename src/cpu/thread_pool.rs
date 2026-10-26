// SPDX-License-Identifier: AGPL-3.0-or-later
// CPU execution model matching llama.cpp

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

/// Default number of threads to use
pub fn default_n_threads() -> usize {
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4)
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

    // Create a reference to f that can be moved into each closure
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

            // Move tid and chunk, borrow f via f_ref (references are Copy)
            s.spawn(move || {
                f_ref(tid, chunk);
            });
        }
    });
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
