// Benchmark tiled GEMV implementation

use infer_train::quant::matmul::k_quant::{
    matmul_q5_k_q8_k, matmul_q5_k_q8_k_single_thread,
};
use infer_train::quant::matmul::tiled::gemv_q5k_q8k_tiled;
use infer_train::quant::types::{BlockQ5K, BlockQ8K, QK_K};
use std::time::Instant;

fn bench(k: usize, m: usize, name: &str) {
    let nb = k / QK_K;

    println!("\n=== {} (K={}, M={}) ===", name, k, m);

    let weights = vec![unsafe { std::mem::zeroed::<BlockQ5K>() }; m * nb];
    let x = vec![unsafe { std::mem::zeroed::<BlockQ8K>() }; nb];
    let mut dst = vec![0.0f32; m];

    // Warm up
    unsafe {
        for _ in 0..10 {
            matmul_q5_k_q8_k_single_thread(&weights, &x, &mut dst, k, m);
        }
    }

    let iterations = 100;

    // Single-thread
    let start = Instant::now();
    unsafe {
        for _ in 0..iterations {
            matmul_q5_k_q8_k_single_thread(&weights, &x, &mut dst, k, m);
        }
    }
    let single_per_iter = start.elapsed().as_nanos() as f64 / iterations as f64;
    println!("Single-thread: {:.2} µs/iter", single_per_iter / 1000.0);

    // Tiled
    let start = Instant::now();
    unsafe {
        for _ in 0..iterations {
            gemv_q5k_q8k_tiled(&weights, &x, &mut dst, k, m);
        }
    }
    let tiled_per_iter = start.elapsed().as_nanos() as f64 / iterations as f64;
    println!("Tiled: {:.2} µs/iter", tiled_per_iter / 1000.0);

    // Multi-thread (hybrid)
    let start = Instant::now();
    unsafe {
        for _ in 0..iterations {
            matmul_q5_k_q8_k(&weights, &x, &mut dst, k, m);
        }
    }
    let multi_per_iter = start.elapsed().as_nanos() as f64 / iterations as f64;
    println!("Multi-thread: {:.2} µs/iter", multi_per_iter / 1000.0);

    println!("Tiled speedup: {:.2}x", single_per_iter / tiled_per_iter);
    println!("Multi speedup: {:.2}x", single_per_iter / multi_per_iter);
}

fn main() {
    // Qwen3-1.5B dimensions
    bench(1536, 1536, "QKV projection");
    bench(1536, 8960, "FFN gate/up");
}
