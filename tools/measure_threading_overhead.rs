// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

//! Measure threading overhead: fixed distribution vs work stealing
//!
//! Simulates the same work distribution pattern as flash attention

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const TOTAL_ROWS: usize = 896; // n_heads * n_tokens for typical prefill
const WORK_PER_ROW_NS: u64 = 100; // Simulated work per row (nanoseconds)

fn main() {
    let n_threads = 10;

    // Method 1: Fixed distribution (our current approach)
    let start = Instant::now();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..n_threads)
            .map(|tid| {
                s.spawn(move || {
                    let rows_per_thread = (TOTAL_ROWS + n_threads - 1) / n_threads;
                    let ir0 = tid * rows_per_thread;
                    let ir1 = (ir0 + rows_per_thread).min(TOTAL_ROWS);

                    // Simulate work
                    let mut sum = 0u64;
                    for i in ir0..ir1 {
                        sum += simulate_work(i);
                    }
                    std::hint::black_box(sum);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    });
    let fixed_time = start.elapsed();

    // Method 2: Work stealing (llama.cpp approach)
    let start = Instant::now();
    use std::sync::Arc;
    let current_chunk = Arc::new(AtomicUsize::new(n_threads)); // First chunk is n_threads (llama.cpp style)
    let chunk_size = 16; // Same as llama.cpp
    let n_chunks = (TOTAL_ROWS + chunk_size - 1) / chunk_size;

    std::thread::scope(|s| {
        for tid in 0..n_threads {
            let current_chunk = current_chunk.clone();
            s.spawn(move || {
                let mut sum = 0u64;
                let mut chunk = tid; // Start with thread ID

                while chunk < n_chunks {
                    let ir0 = chunk * chunk_size;
                    let ir1 = (ir0 + chunk_size).min(TOTAL_ROWS);

                    for i in ir0..ir1 {
                        sum += simulate_work(i);
                    }

                    // Work stealing: get next chunk
                    chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
                }
                std::hint::black_box(sum);
            });
        }
    });
    let stealing_time = start.elapsed();

    println!("Threading overhead comparison:");
    println!("  Fixed distribution: {:?}", fixed_time);
    println!("  Work stealing:      {:?}", stealing_time);
    println!(
        "  Ratio:              {:.2}x",
        fixed_time.as_nanos() as f64 / stealing_time.as_nanos() as f64
    );

    // Measure thread creation overhead alone
    let start = Instant::now();
    for _ in 0..10 {
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..n_threads).map(|_| s.spawn(|| {})).collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }
    let thread_creation_time = start.elapsed() / 10;
    println!("\nThread creation overhead: {:?}", thread_creation_time);
}

#[inline(never)]
fn simulate_work(row: usize) -> u64 {
    // Simulate ~100ns of work per row
    let start = Instant::now();
    let mut sum = row as u64;
    while start.elapsed().as_nanos() < WORK_PER_ROW_NS as u128 {
        sum = sum.wrapping_add(1);
        std::hint::black_box(sum);
    }
    sum
}
