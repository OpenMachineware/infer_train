// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Rust F16 GEMM benchmark
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
    println!("=== Rust F16 GEMM Performance ===\n");

    let tests: [(usize, usize, usize); 3] =
        [(128, 16, 512), (512, 32, 512), (1024, 64, 1024)];

    unsafe {
        for &(m, n, k) in &tests {
            println!("M={}, N={}, K={}", m, n, k);

            // Create test data
            let a: Vec<f16> = (0..m * k)
                .map(|i| f16::from_f32(((i % 17) as f32) * 0.1))
                .collect();
            let b: Vec<f16> = (0..n * k)
                .map(|i| f16::from_f32(((i % 13) as f32) * 0.1))
                .collect();
            let mut c = vec![0.0f32; n * m];
            let mut c_ref = vec![0.0f32; n * m];

            // Verify correctness
            reference_hgemm(&a, &b, &mut c_ref, m, n, k);
            hgemm(&a, &b, &mut c, m, n, k);

            let mut max_error = 0.0f32;
            for i in 0..(n * m) {
                let err = (c_ref[i] - c[i]).abs();
                if err > max_error {
                    max_error = err;
                }
            }
            println!("  Max error: {:.2e}", max_error);

            // Warmup
            for _ in 0..10 {
                hgemm(&a, &b, &mut c, m, n, k);
            }

            // Benchmark
            let iterations = 1000;
            let start = Instant::now();
            for _ in 0..iterations {
                hgemm(&a, &b, &mut c, m, n, k);
            }
            let elapsed_us = start.elapsed().as_secs_f64() * 1e6;

            let avg_us = elapsed_us / (iterations as f64);
            let gflops = (2.0 * (m as f64) * (n as f64) * (k as f64))
                / (avg_us * 1e-6)
                / 1e9;

            println!("  Avg time: {:.2} us", avg_us);
            println!("  GFLOPS:   {:.2}\n", gflops);
        }
    }
}
