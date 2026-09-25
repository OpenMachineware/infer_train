// Benchmark RMSNorm performance
// Run: cargo run --release --bin benchmark_rms_norm

use infer_train::ops::rms_norm::rms_norm_f32;
use std::time::Instant;

fn main() {
    println!("=== RMSNorm Performance Benchmark ===\n");

    let hidden_dim = 4096;
    let n_rows = 1;
    let eps = 1e-5;
    let n_iterations = 10000;

    // Create test data
    let x: Vec<f32> = (0..n_rows * hidden_dim)
        .map(|i| (i % 100) as f32 * 0.01 - 0.5)
        .collect();

    let w: Vec<f32> = (0..hidden_dim).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect();

    let mut dst = vec![0.0f32; n_rows * hidden_dim];

    // Warmup
    for _ in 0..100 {
        unsafe {
            rms_norm_f32(&x, &w, &mut dst, hidden_dim, eps);
        }
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..n_iterations {
        unsafe {
            rms_norm_f32(&x, &w, &mut dst, hidden_dim, eps);
        }
    }
    let elapsed = start.elapsed();

    let total_ops = n_iterations as f64 * hidden_dim as f64 * 3.0; // 3 ops per element: square, multiply, multiply
    let gflops = total_ops / elapsed.as_secs_f64() / 1e9;

    let time_per_iter = elapsed.as_micros() as f64 / n_iterations as f64;
    let time_per_row = time_per_iter / n_rows as f64;

    println!("Hidden dim: {}", hidden_dim);
    println!("N rows: {}", n_rows);
    println!("Iterations: {}", n_iterations);
    println!("Time per iteration: {:.2} us", time_per_iter);
    println!("Time per row: {:.2} us", time_per_row);
    println!("GFLOPS: {:.2}", gflops);
    println!();

    // Test different hidden dimensions
    println!("=== Hidden Dimension Scaling ===\n");
    let dims = [512, 1024, 2048, 4096, 8192];

    for dim in dims {
        let x: Vec<f32> = (0..dim).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect();
        let mut dst = vec![0.0f32; dim];

        // Warmup
        for _ in 0..100 {
            unsafe {
                rms_norm_f32(&x, &w, &mut dst, dim, eps);
            }
        }

        let iterations = (100000 / (dim / 512)).max(1000);
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                rms_norm_f32(&x, &w, &mut dst, dim, eps);
            }
        }
        let elapsed = start.elapsed();

        let total_ops = iterations as f64 * dim as f64 * 3.0;
        let gflops = total_ops / elapsed.as_secs_f64() / 1e9;
        let time_us = elapsed.as_micros() as f64 / iterations as f64;

        println!(
            "dim={:5}: {:.2} us, {:.2} GFLOPS",
            dim, time_us, gflops
        );
    }

    // Test batch processing
    println!("\n=== Batch Processing ===\n");
    let batch_sizes = [1, 4, 16, 64, 256];
    let dim = 4096;

    for batch in batch_sizes {
        let x: Vec<f32> = (0..batch * dim)
            .map(|i| (i % 100) as f32 * 0.01 - 0.5)
            .collect();
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect();
        let mut dst = vec![0.0f32; batch * dim];

        // Warmup
        for _ in 0..10 {
            unsafe {
                rms_norm_f32(&x, &w, &mut dst, dim, eps);
            }
        }

        let iterations = (10000 / batch).max(100);
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                rms_norm_f32(&x, &w, &mut dst, dim, eps);
            }
        }
        let elapsed = start.elapsed();

        let total_elements = iterations as f64 * batch as f64 * dim as f64;
        let total_ops = total_elements * 3.0;
        let gflops = total_ops / elapsed.as_secs_f64() / 1e9;
        let time_us = elapsed.as_micros() as f64 / iterations as f64;

        println!(
            "batch={:3}: {:.2} us total, {:.2} us/row, {:.2} GFLOPS",
            batch, time_us, time_us / batch as f64, gflops
        );
    }
}
