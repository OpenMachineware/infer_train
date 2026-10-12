// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Verify F16 GEMM correctness
use half::f16;
use infer_train::quant::matmul::*;
use std::time::Instant;

fn reference_hgemm(
    a: &[f16],
    b: &[f16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l].to_f32() * b[j * k + l].to_f32();
            }
            c[j * m + i] = sum;
        }
    }
}

fn main() {
    println!("=== F16 GEMM Correctness Test ===\n");

    // Test various sizes
    let test_cases = [
        (4, 8, 32),    // Small
        (16, 16, 128), // Medium
        (64, 16, 512), // Prefill-like (N=16 for batch)
        (128, 8, 512), // Prefill (N=8)
    ];

    unsafe {
        for &(m, n, k) in &test_cases {
            // Generate test data with balanced range
            let a: Vec<f16> = (0..m * k)
                .map(|i| {
                    f16::from_f32(
                        ((i % 17) as f32 * 0.1 - 0.8).clamp(-1.0, 1.0),
                    )
                })
                .collect();
            let b: Vec<f16> = (0..n * k)
                .map(|i| {
                    f16::from_f32(
                        ((i % 13) as f32 * 0.1 - 0.6).clamp(-1.0, 1.0),
                    )
                })
                .collect();

            // Reference result
            let mut c_ref = vec![0.0f32; n * m];
            reference_hgemm(&a, &b, &mut c_ref, m, n, k);

            // Our implementation
            let mut c_ours = vec![0.0f32; n * m];
            hgemm(&a, &b, &mut c_ours, m, n, k);

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

            let status = if max_error < 2e-2 { "✓" } else { "✗" };
            println!(
                "{} M={}, N={}, K={}: max error = {:.2e} at ({}, {})",
                status, m, n, k, max_error, max_idx.0, max_idx.1
            );
        }

        // Performance comparison
        println!("\n=== F16 GEMM Performance ===\n");

        let (m, n, k) = (128, 16, 512);
        let a: Vec<f16> =
            (0..m * k).map(|i| f16::from_f32((i % 17) as f32 * 0.1)).collect();
        let b: Vec<f16> =
            (0..n * k).map(|i| f16::from_f32((i % 13) as f32 * 0.1)).collect();
        let mut c = vec![0.0f32; n * m];

        // Warmup
        for _ in 0..10 {
            hgemm(&a, &b, &mut c, m, n, k);
        }

        // Benchmark
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            hgemm(&a, &b, &mut c, m, n, k);
        }
        let elapsed = start.elapsed().as_secs_f64();

        let flops =
            2.0 * (m as f64) * (n as f64) * (k as f64) * (iterations as f64);
        let gflops = flops / elapsed / 1e9;

        println!("F16 GEMM {}x{}x{}: {:.2} GFLOPS", m, n, k, gflops);

        // Compare with FP32
        let a_f32: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();
        let mut c_f32 = vec![0.0f32; n * m];

        let start = Instant::now();
        for _ in 0..iterations {
            sgemm_optimized(&a_f32, &b_f32, &mut c_f32, m, n, k);
        }
        let elapsed_f32 = start.elapsed().as_secs_f64();
        let gflops_f32 = flops / elapsed_f32 / 1e9;

        println!("FP32 GEMM {}x{}x{}: {:.2} GFLOPS", m, n, k, gflops_f32);
        println!("F16 speedup: {:.2}x", gflops / gflops_f32);
    }
}
