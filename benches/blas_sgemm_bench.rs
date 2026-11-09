// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Jia Liu

// Test Apple Accelerate BLAS performance

use std::ffi::c_int;
use std::time::Instant;

// Apple Accelerate BLAS bindings
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    pub fn cblas_sgemm(
        order: u32,  // CblasRowMajor or CblasColMajor
        transa: u32, // CblasNoTrans, CblasTrans, CblasConjTrans
        transb: u32,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        c: *mut f32,
        ldc: c_int,
    );
}

const CBLAS_ROW_MAJOR: u32 = 101;
const CBLAS_NO_TRANS: u32 = 111;
const CBLAS_TRANS: u32 = 112;

fn bench_neon_gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    iterations: usize,
) -> f64 {
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();

        // Simple NEON-style row-major GEMM (scalar for now, should use our optimized version)
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }

        times.push(start.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[iterations / 2]
}

fn bench_blas_sgemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    iterations: usize,
) -> f64 {
    let mut times = Vec::with_capacity(iterations);

    // We want: C(m×n) = A(m×k) × B(k×n)
    // BLAS expects column-major by default, but we have row-major data.
    //
    // Using row-major layout:
    // A: m×k, lda = k
    // B: k×n (but we'll transpose it), ldb = n, leading dimension after transpose = k
    // C: m×n, ldc = n
    //
    // cblas_sgemm(row-major, NoTrans, Trans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // computes: C = alpha * A * B^T + beta * C
    // where B is treated as having ldb columns (before transpose)

    for _ in 0..iterations {
        let start = Instant::now();

        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS, // A: m×k
                CBLAS_NO_TRANS, // B: k×n (not transposed)
                m as c_int,
                n as c_int,
                k as c_int,
                1.0,
                a.as_ptr(),
                k as c_int, // lda = k (A has k columns per row)
                b.as_ptr(),
                n as c_int, // ldb = n (B has n columns per row)
                0.0,
                c.as_mut_ptr(),
                n as c_int, // ldc = n (C has n columns per row)
            );
        }

        times.push(start.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[iterations / 2]
}

fn main() {
    let m = 4096;
    let k = 4096;
    let n = 128;
    let iterations = 30;

    println!("=== Apple Accelerate BLAS vs Scalar GEMM ===");
    println!("M={}, K={}, N={}, {} iterations\n", m, k, n, iterations);

    // Generate test data (row-major)
    let a: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 100) as f32 * 0.01).collect();
    let mut c_scalar = vec![0.0f32; m * n];
    let mut c_blas = vec![0.0f32; m * n];

    // Benchmark scalar
    println!("Benchmarking scalar GEMM...");
    let scalar_time =
        bench_neon_gemm(&a, &b, &mut c_scalar, m, n, k, iterations);

    // Benchmark BLAS
    println!("Benchmarking BLAS sgemm...");
    let blas_time = bench_blas_sgemm(&a, &b, &mut c_blas, m, n, k, iterations);

    // Verify correctness (spot check)
    let mut max_diff = 0.0f32;
    for i in 0..m * n {
        max_diff = max_diff.max((c_scalar[i] - c_blas[i]).abs());
    }

    let scalar_ms = scalar_time / 1e6;
    let blas_ms = blas_time / 1e6;
    let speedup = scalar_time / blas_time;

    let flops = 2.0 * m as f64 * k as f64 * n as f64;
    let scalar_gflops = flops / (scalar_time / 1e9) / 1e9;
    let blas_gflops = flops / (blas_time / 1e9) / 1e9;

    println!("\n=== Results ===");
    println!("Scalar:      {:8.2} ms ({:.1} GFLOPS)", scalar_ms, scalar_gflops);
    println!("BLAS:        {:8.2} ms ({:.1} GFLOPS)", blas_ms, blas_gflops);
    println!("Speedup:     {:8.2}x", speedup);
    println!(
        "Correctness: max_diff = {:.6} ({})",
        max_diff,
        if max_diff < 1e-2 { "OK" } else { "FAIL" }
    );

    println!("\n=== Conclusion ===");
    if speedup > 2.0 {
        println!("BLAS is {:.1}x faster - worth integrating!", speedup);
    } else {
        println!(
            "BLAS is only {:.1}x faster - may not be worth the complexity",
            speedup
        );
    }
}
