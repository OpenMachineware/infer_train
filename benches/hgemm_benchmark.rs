// CPU GEMM benchmark: FP32 vs F16 performance
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
    println!("=== CPU GEMM: FP32 vs F16 Performance ===\n");

    let test_cases = [
        (128, 16, 512),   // Small batch prefill
        (512, 32, 512),   // Medium batch
        (1024, 64, 1024), // Large batch
    ];

    unsafe {
        for &(m, n, k) in &test_cases {
            println!("M={}, N={}, K={}", m, n, k);

            // FP32 data
            let a_f32: Vec<f32> =
                (0..m * k).map(|i| (i % 17) as f32 * 0.1).collect();
            let b_f32: Vec<f32> =
                (0..n * k).map(|i| (i % 13) as f32 * 0.1).collect();
            let mut c_f32 = vec![0.0f32; n * m];

            // F16 data
            let a_f16: Vec<f16> =
                a_f32.iter().map(|&x| f16::from_f32(x)).collect();
            let b_f16: Vec<f16> =
                b_f32.iter().map(|&x| f16::from_f32(x)).collect();
            let mut c_f16 = vec![0.0f32; n * m];

            // Warmup
            for _ in 0..10 {
                sgemm_optimized(&a_f32, &b_f32, &mut c_f32, m, n, k);
                hgemm(&a_f16, &b_f16, &mut c_f16, m, n, k);
            }

            // Benchmark
            let iterations = if m > 1000 { 20 } else { 100 };

            // FP32
            let start = Instant::now();
            for _ in 0..iterations {
                sgemm_optimized(&a_f32, &b_f32, &mut c_f32, m, n, k);
            }
            let fp32_time = start.elapsed().as_secs_f64();
            let fp32_gflops = 2.0
                * (m as f64)
                * (n as f64)
                * (k as f64)
                * (iterations as f64)
                / fp32_time
                / 1e9;

            // F16
            let start = Instant::now();
            for _ in 0..iterations {
                hgemm(&a_f16, &b_f16, &mut c_f16, m, n, k);
            }
            let f16_time = start.elapsed().as_secs_f64();
            let f16_gflops = 2.0
                * (m as f64)
                * (n as f64)
                * (k as f64)
                * (iterations as f64)
                / f16_time
                / 1e9;

            // Verify F16 precision
            let mut max_error = 0.0f32;
            for i in 0..(n * m) {
                let error = (c_f32[i] - c_f16[i]).abs();
                max_error = max_error.max(error);
            }

            let speedup = f16_gflops / fp32_gflops;
            println!("  FP32 GEMM:   {:.2} GFLOPS", fp32_gflops);
            println!("  F16 GEMM:    {:.2} GFLOPS", f16_gflops);
            println!("  F16 speedup: {:.2}x", speedup);
            println!("  Max error:   {:.2e}", max_error);
            println!();
        }
    }

    println!("F16 GEMM is faster due to:");
    println!("  - Half memory bandwidth (2x less data to load)");
    println!("  - F16 FMLA has same throughput as F32 FMLA on Apple M1");
    println!("  - Better cache utilization");
}
