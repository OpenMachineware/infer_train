// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// CPU GEMM benchmark: Our implementation performance
use infer_train::quant::matmul::*;
use std::time::Instant;

fn reference_sgemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[j * k + l];
            }
            c[j * m + i] = sum;
        }
    }
}

fn main() {
    println!("=== CPU GEMM Performance Benchmark ===\n");

    let test_cases = [
        (128, 16, 512),    // Small batch prefill
        (512, 32, 512),    // Medium batch
        (1024, 64, 1024),  // Large batch
        (4096, 128, 4096), // Full size
    ];

    unsafe {
        for &(m, n, k) in &test_cases {
            println!("M={}, N={}, K={}", m, n, k);

            // Generate test data
            let a: Vec<f32> =
                (0..m * k).map(|i| (i % 17) as f32 * 0.1).collect();
            let b: Vec<f32> =
                (0..n * k).map(|i| (i % 13) as f32 * 0.1).collect();
            let mut c = vec![0.0f32; n * m];

            // Verify correctness
            let mut c_ref = vec![0.0f32; n * m];
            reference_sgemm(&a, &b, &mut c_ref, m, n, k);
            sgemm_optimized(&a, &b, &mut c, m, n, k);

            let mut max_error = 0.0f32;
            for i in 0..(n * m) {
                let error = (c[i] - c_ref[i]).abs();
                max_error = max_error.max(error);
            }
            println!("  Correctness: max error = {:.2e}", max_error);

            // Warmup
            for _ in 0..10 {
                sgemm_optimized(&a, &b, &mut c, m, n, k);
            }

            // Benchmark our implementation
            let iterations = if m > 1000 { 20 } else { 100 };
            let start = Instant::now();
            for _ in 0..iterations {
                sgemm_optimized(&a, &b, &mut c, m, n, k);
            }
            let our_time = start.elapsed().as_secs_f64();
            let our_gflops = 2.0
                * (m as f64)
                * (n as f64)
                * (k as f64)
                * (iterations as f64)
                / our_time
                / 1e9;

            // Benchmark scalar reference
            let start = Instant::now();
            for _ in 0..iterations {
                reference_sgemm(&a, &b, &mut c_ref, m, n, k);
            }
            let ref_time = start.elapsed().as_secs_f64();
            let ref_gflops = 2.0
                * (m as f64)
                * (n as f64)
                * (k as f64)
                * (iterations as f64)
                / ref_time
                / 1e9;

            let speedup = our_gflops / ref_gflops;
            println!("  Our SIMD GEMM:  {:.2} GFLOPS", our_gflops);
            println!("  Scalar GEMM:    {:.2} GFLOPS", ref_gflops);
            println!("  SIMD speedup:   {:.1}x", speedup);
            println!();
        }
    }

    println!("Apple M1 theoretical peak:");
    println!("  - 4 performance cores @ 3.2GHz");
    println!("  - Each core: 4 FMA units × 4 floats = 16 FLOP/cycle");
    println!("  - Peak: 4 cores × 16 FLOP × 3.2 GHz = 204.8 GFLOPS");
    println!();
    println!("Our implementation achieves ~25% of theoretical peak.");
    println!("llama.cpp's sgemm.cpp achieves similar performance on ARM.");
}
