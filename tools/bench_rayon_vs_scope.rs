// Benchmark Rayon persistent thread pool vs thread::scope ephemeral threads

use rayon::prelude::*;
use std::thread;
use std::time::Instant;

fn main() {
    let n_threads = 10;
    let n_iterations = 1000;
    let work_size = 11008; // Typical intermediate_size

    println!("=== Thread Pool Comparison Benchmark ===");
    println!(
        "n_threads: {}, iterations: {}, work_size: {}",
        n_threads, n_iterations, work_size
    );

    // Create test data
    let data: Vec<f32> = vec![1.0; work_size];
    let mut output: Vec<f32> = vec![0.0; work_size];

    // Warm up Rayon
    (0..10).into_par_iter().for_each(|_| {});

    // Benchmark 1: thread::scope (ephemeral threads)
    let mut total_scope_time = 0u64;
    for _ in 0..n_iterations {
        let start = Instant::now();

        let chunk_size = (work_size + n_threads - 1) / n_threads;
        thread::scope(|s| {
            let mut remaining = output.as_mut_slice();
            let mut tid = 0;
            while !remaining.is_empty() {
                let split_point = chunk_size.min(remaining.len());
                let (chunk, rest) = remaining.split_at_mut(split_point);
                remaining = rest;
                let src = &data;
                s.spawn(move || {
                    for (i, out) in chunk.iter_mut().enumerate() {
                        *out = src[tid * chunk_size + i] * 2.0;
                    }
                });
                tid += 1;
            }
        });

        total_scope_time += start.elapsed().as_nanos() as u64;
    }

    // Benchmark 2: Rayon (persistent thread pool)
    let mut total_rayon_time = 0u64;
    for _ in 0..n_iterations {
        let start = Instant::now();

        let src = &data;
        output.par_iter_mut().enumerate().for_each(|(i, out)| {
            *out = src[i] * 2.0;
        });

        total_rayon_time += start.elapsed().as_nanos() as u64;
    }

    // Results
    println!("\nthread::scope (ephemeral threads):");
    println!("  Total: {:.2} ms", total_scope_time as f64 / 1_000_000.0);
    println!(
        "  Per iteration: {:.2} µs",
        total_scope_time as f64 / n_iterations as f64 / 1000.0
    );

    println!("\nRayon (persistent thread pool):");
    println!("  Total: {:.2} ms", total_rayon_time as f64 / 1_000_000.0);
    println!(
        "  Per iteration: {:.2} µs",
        total_rayon_time as f64 / n_iterations as f64 / 1000.0
    );

    let speedup = total_scope_time as f64 / total_rayon_time as f64;
    println!("\nSpeedup: {:.2}x", speedup);

    // Estimate for forward pass
    let n_gemv_per_layer = 7;
    let n_layers = 32;
    let total_gemv = n_gemv_per_layer * n_layers;
    let scope_overhead = (total_scope_time - total_rayon_time) as f64
        / n_iterations as f64
        / 1000.0;

    println!("\n=== Estimated per forward pass ===");
    println!(
        "Thread creation savings: {:.2} ms",
        scope_overhead * total_gemv as f64 / 1000.0
    );
}
