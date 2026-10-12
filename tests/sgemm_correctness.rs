// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Verify GEMM correctness
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
    println!("=== GEMM Correctness Test ===\n");

    // Test various sizes
    let test_cases = [
        (4, 4, 16),    // Small
        (16, 16, 64),  // Medium
        (64, 64, 256), // Large
        (128, 8, 512), // Prefill-like
        (8, 128, 512), // Transposed
    ];

    unsafe {
        for &(m, n, k) in &test_cases {
            // Generate test data
            let a: Vec<f32> =
                (0..m * k).map(|i| (i % 17) as f32 * 0.1).collect();
            let b: Vec<f32> =
                (0..n * k).map(|i| (i % 13) as f32 * 0.1).collect();

            // Reference result
            let mut c_ref = vec![0.0f32; n * m];
            reference_sgemm(&a, &b, &mut c_ref, m, n, k);

            // Our implementation
            let mut c_ours = vec![0.0f32; n * m];
            sgemm_optimized(&a, &b, &mut c_ours, m, n, k);

            // Check error
            let mut max_error = 0.0f32;
            let mut max_idx = (0, 0);
            for j in 0..n {
                for i in 0..m {
                    let error = (c_ref[j * m + i] - c_ours[j * m + i]).abs();
                    if error > max_error {
                        max_error = error;
                        max_idx = (i, j);
                    }
                }
            }

            let status = if max_error < 1e-4 { "✓" } else { "✗" };
            println!(
                "{} M={}, N={}, K={}: max error = {:.2e} at ({}, {})",
                status, m, n, k, max_error, max_idx.0, max_idx.1
            );
        }

        // Performance comparison
        println!("\n=== Performance Comparison ===\n");

        let (m, n, k) = (128, 128, 512);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i % 13) as f32 * 0.1).collect();
        let mut c = vec![0.0f32; n * m];

        // Warmup
        for _ in 0..10 {
            sgemm_optimized(&a, &b, &mut c, m, n, k);
        }

        // Benchmark
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            sgemm_optimized(&a, &b, &mut c, m, n, k);
        }
        let elapsed = start.elapsed().as_secs_f64();

        let flops =
            2.0 * (m as f64) * (n as f64) * (k as f64) * (iterations as f64);
        let gflops = flops / elapsed / 1e9;

        println!("FP32 GEMM {}x{}x{}: {:.2} GFLOPS", m, n, k, gflops);
    }
}
