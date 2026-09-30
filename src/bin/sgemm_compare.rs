// Rust GEMM comparison test - must use same data as C++ version
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

fn get_time_us() -> f64 {
    Instant::now().elapsed().as_secs_f64() * 1e6
}

fn main() {
    println!("=== Rust GEMM Performance ===\n");

    let tests: [(usize, usize, usize); 3] =
        [(128, 16, 512), (512, 32, 512), (1024, 64, 1024)];

    unsafe {
        for &(m, n, k) in &tests {
            println!("M={}, N={}, K={}", m, n, k);

            // Same data as C++ version
            let a: Vec<f32> =
                (0..m * k).map(|i| ((i % 17) as f32) * 0.1).collect();
            let b: Vec<f32> =
                (0..n * k).map(|i| ((i % 13) as f32) * 0.1).collect();
            let mut c = vec![0.0f32; n * m];
            let mut c_ref = vec![0.0f32; n * m];

            // Verify correctness
            reference_sgemm(&a, &b, &mut c_ref, m, n, k);
            sgemm_optimized(&a, &b, &mut c, m, n, k);

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
                sgemm_optimized(&a, &b, &mut c, m, n, k);
            }

            // Benchmark
            let iterations = 1000; // Use many iterations for all sizes
            let start = Instant::now();
            for _ in 0..iterations {
                sgemm_optimized(&a, &b, &mut c, m, n, k);
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

    println!("Compare with C++:\n");
    println!("  cd llama.cpp-0.5.0");
    println!("  g++ -O3 -mcpu=native test_sgemm_simple.cpp -o test_sgemm_cpp");
    println!("  ./test_sgemm_cpp");
}
