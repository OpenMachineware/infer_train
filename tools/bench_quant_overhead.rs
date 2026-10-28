// Benchmark quantization overhead vs GEMV computation

use infer_train::quant::types::{BlockQ8K, QK_K};
use infer_train::quant::kv_cache::quantize_row_q8_k_neon;

fn main() {
    let k = 1024;  // typical hidden size
    let nb = k / QK_K;

    // Create test input
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut y = vec![BlockQ8K::default(); nb];

    // Benchmark quantization
    let n_iterations = 1000;
    let start = std::time::Instant::now();
    for _ in 0..n_iterations {
        quantize_row_q8_k_neon(&x, &mut y);
    }
    let quant_time = start.elapsed().as_micros() as f64 / n_iterations as f64;

    println!("Quantization time (K=1024): {:.2} µs", quant_time);
    println!("Per-block time: {:.2} µs", quant_time / nb as f64);

    // Estimate total quantization overhead per forward pass
    // 28 layers × 7 GEMV per layer = 196 quantizations
    // Plus quantization sizes vary (1024, 2048, 3072)
    let total_quant_overhead_us = quant_time * 196.0;
    println!("Estimated total quantization overhead per forward: {:.2} µs ({:.2} ms)",
        total_quant_overhead_us, total_quant_overhead_us / 1000.0);
}
