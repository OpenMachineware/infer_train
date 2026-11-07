// Measure GEMM/GEMV kernel performance in isolation
use infer_train::quant::types::{BlockQ5K, BlockQ8K, QK_K};
use std::time::Instant;

fn main() {
    println!("=== GEMM/GEMV Kernel Performance ===\n");

    // Test dimensions matching Qwen3-0.6B
    let k = 1024;  // hidden_size
    let m_values = [2048, 8960, 3072]; // QKV, FFN gate/up, combined QKV
    let n_values = [1, 16, 32, 64]; // batch sizes

    let nb = k / QK_K;

    for &m in &m_values {
        println!("--- m = {} (k = {}) ---", m, k);
        for &n in &n_values {
            // Create test data
            let weights: Vec<BlockQ5K> = vec![unsafe { std::mem::zeroed() }; m * nb];
            let inputs: Vec<BlockQ8K> = vec![unsafe { std::mem::zeroed() }; n * nb];
            let mut outputs: Vec<f32> = vec![0.0; n * m];

            // Warmup
            for _ in 0..3 {
                unsafe {
                    infer_train::quant::matmul::tiled::matmul_q5k_q8k_tiled(
                        &weights, &inputs, &mut outputs, k, m, n
                    );
                }
            }

            // Benchmark
            let iters = if n == 1 { 100 } else { 20 };
            let start = Instant::now();
            for _ in 0..iters {
                unsafe {
                    infer_train::quant::matmul::tiled::matmul_q5k_q8k_tiled(
                        &weights, &inputs, &mut outputs, k, m, n
                    );
                }
            }
            let elapsed = start.elapsed();

            // Calculate metrics
            let flops = 2.0 * m as f64 * k as f64 * n as f64 * iters as f64;
            let gflops = flops / elapsed.as_secs_f64() / 1e9;
            let ms_per_call = elapsed.as_secs_f64() * 1000.0 / iters as f64;

            println!("  n={}: {:.2} ms/call, {:.1} GFLOPS", n, ms_per_call, gflops);
        }
    }

    println!("\n=== Performance Target ===");
    println!("llama.cpp achieves ~200 GFLOPS for quantized GEMM on M4");
    println!("Memory bandwidth ~100 GB/s, quantized GEMM should be compute-bound");
}
