// Benchmark lock-free persistent thread pool for GEMV
// SPDX-License-Identifier: AGPL-3.0-or-later

use infer_train::cpu::default_n_threads;
use infer_train::quant::matmul::k_quant::{
    matmul_q5_k_q8_k, matmul_q5_k_q8_k_hybrid, matmul_q5_k_q8_k_single_thread,
};
use infer_train::quant::types::{BlockQ5K, BlockQ8K, QK_K};
use std::time::Instant;

fn main() {
    println!("=== Lock-Free Thread Pool GEMV Benchmark ===\n");

    let n_threads = default_n_threads();
    println!("Threads: {}", n_threads);

    // Create test data (realistic sizes for decode)
    let k = 4096; // hidden_size
    let m_values = [256, 512, 1024, 2048, 4096, 8192]; // output sizes
    let n_iterations = 100;

    for m in m_values {
        let nb = k / QK_K;

        // Allocate weights and input (zeroed bytes)
        let weights: Vec<BlockQ5K> =
            vec![unsafe { std::mem::zeroed() }; m * nb];
        let input: Vec<BlockQ8K> = vec![unsafe { std::mem::zeroed() }; nb];
        let mut output: Vec<f32> = vec![0.0; m];
        let mut output2: Vec<f32> = vec![0.0; m];
        let mut output3: Vec<f32> = vec![0.0; m];

        println!("\n--- M = {} rows ---", m);

        // Warmup
        for _ in 0..10 {
            unsafe {
                matmul_q5_k_q8_k_hybrid(&weights, &input, &mut output, k, m);
            }
        }

        // Test 1: Single-thread
        let start = Instant::now();
        for _ in 0..n_iterations {
            unsafe {
                matmul_q5_k_q8_k_single_thread(
                    &weights,
                    &input,
                    &mut output,
                    k,
                    m,
                );
            }
        }
        let st_time = start.elapsed();
        let st_us = st_time.as_micros() as f64 / n_iterations as f64;
        println!("Single-thread: {:.2} µs/iter", st_us);

        // Test 2: std::thread::scope (parallel_chunks)
        let start = Instant::now();
        for _ in 0..n_iterations {
            unsafe {
                matmul_q5_k_q8_k(&weights, &input, &mut output2, k, m);
            }
        }
        let scope_time = start.elapsed();
        let scope_us = scope_time.as_micros() as f64 / n_iterations as f64;
        println!("std::thread::scope: {:.2} µs/iter", scope_us);

        // Test 3: Hybrid ST/MT with lock-free pool
        let start = Instant::now();
        for _ in 0..n_iterations {
            unsafe {
                matmul_q5_k_q8_k_hybrid(&weights, &input, &mut output3, k, m);
            }
        }
        let hybrid_time = start.elapsed();
        let hybrid_us = hybrid_time.as_micros() as f64 / n_iterations as f64;
        println!("Hybrid (lock-free pool): {:.2} µs/iter", hybrid_us);

        // Speedup
        if st_us > 0.0 && hybrid_us > 0.0 {
            let speedup_vs_st = st_us / hybrid_us;
            let speedup_vs_scope = scope_us / hybrid_us;
            println!("Speedup vs ST: {:.2}x", speedup_vs_st);
            println!("Speedup vs scope: {:.2}x", speedup_vs_scope);
        }
    }
}
