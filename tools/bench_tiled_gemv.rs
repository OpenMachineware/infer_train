// Benchmark tiled GEMV implementation

use half::f16;
use infer_train::quant::matmul::k_quant::{
    matmul_q5_k_q8_k, matmul_q5_k_q8_k_single_thread,
};
use infer_train::quant::matmul::tiled::gemv_q5k_q8k_tiled;
use infer_train::quant::types::{BlockQ5K, BlockQ8K, QK_K};
use std::time::Instant;

fn main() {
    // Test dimensions matching Qwen3-0.6B GEMV
    let k = 896; // hidden_size
    let m = 896; // output dimension (e.g., q_proj)
    let nb = k / QK_K;

    println!("=== Tiled GEMV Benchmark ===");
    println!("K = {} ({} blocks), M = {}", k, nb, m);

    // Create test data - use zeroed blocks
    // For correctness test, we'd need real quantized data
    let weights = vec![unsafe { std::mem::zeroed::<BlockQ5K>() }; m * nb];
    let x = vec![unsafe { std::mem::zeroed::<BlockQ8K>() }; nb];
    let mut dst = vec![0.0f32; m];

    // Warm up
    unsafe {
        for _ in 0..10 {
            matmul_q5_k_q8_k_single_thread(&weights, &x, &mut dst, k, m);
        }
    }

    // Benchmark single-thread
    let iterations = 1000;
    let start = Instant::now();
    unsafe {
        for _ in 0..iterations {
            matmul_q5_k_q8_k_single_thread(&weights, &x, &mut dst, k, m);
        }
    }
    let single_time = start.elapsed();
    let single_per_iter = single_time.as_nanos() as f64 / iterations as f64;
    println!(
        "\nSingle-thread: {:?} ({:.2} µs/iter)",
        single_time,
        single_per_iter / 1000.0
    );

    // Benchmark tiled
    let start = Instant::now();
    unsafe {
        for _ in 0..iterations {
            gemv_q5k_q8k_tiled(&weights, &x, &mut dst, k, m);
        }
    }
    let tiled_time = start.elapsed();
    let tiled_per_iter = tiled_time.as_nanos() as f64 / iterations as f64;
    println!(
        "Tiled: {:?} ({:.2} µs/iter)",
        tiled_time,
        tiled_per_iter / 1000.0
    );

    // Speedup
    let speedup = single_per_iter / tiled_per_iter;
    println!("Speedup: {:.2}x", speedup);

    // Benchmark multi-thread
    let start = Instant::now();
    unsafe {
        for _ in 0..iterations {
            matmul_q5_k_q8_k(&weights, &x, &mut dst, k, m);
        }
    }
    let multi_time = start.elapsed();
    let multi_per_iter = multi_time.as_nanos() as f64 / iterations as f64;
    println!(
        "\nMulti-thread: {:?} ({:.2} µs/iter)",
        multi_time,
        multi_per_iter / 1000.0
    );

    // Compare with single-thread
    let multi_speedup = single_per_iter / multi_per_iter;
    println!("Multi-thread speedup vs single: {:.2}x", multi_speedup);
}
