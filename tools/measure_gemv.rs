// Measure GEMM/GEMV kernel performance in isolation
use infer_train::quant::types::{BlockQ5K, BlockQ8K, QK_K};
use infer_train::model::tensor::{DataType, Tensor};
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
            // Create test data for old tiled implementation
            let weights: Vec<BlockQ5K> = vec![unsafe { std::mem::zeroed() }; m * nb];
            let inputs: Vec<BlockQ8K> = vec![unsafe { std::mem::zeroed() }; n * nb];
            let mut outputs: Vec<f32> = vec![0.0; n * m];

            // Test old tiled implementation
            let iters = if n == 1 { 100 } else { 20 };
            let start = Instant::now();
            for _ in 0..iters {
                unsafe {
                    infer_train::quant::matmul::tiled::matmul_q5k_q8k_tiled(
                        &weights, &inputs, &mut outputs, k, m, n
                    );
                }
            }
            let tiled_elapsed = start.elapsed();
            let flops = 2.0 * m as f64 * k as f64 * n as f64 * iters as f64;
            let tiled_gflops = flops / tiled_elapsed.as_secs_f64() / 1e9;
            let tiled_ms = tiled_elapsed.as_secs_f64() * 1000.0 / iters as f64;

            // Test new unified dispatcher (gemm_dispatch)
            // Create Tensor wrapper for weights
            let weights_tensor = Tensor {
                bytes: bytemuck::cast_slice(&weights).to_vec(),
                shape: vec![k, m], // [K, M] in GGUF
                dtype: DataType::Q5_K,
            };

            // Create F32 inputs for gemm_dispatch (it handles quantization internally)
            let inputs_f32: Vec<f32> = vec![0.0; n * k];
            let mut outputs_new: Vec<f32> = vec![0.0; n * m];

            let start = Instant::now();
            for _ in 0..iters {
                infer_train::quant::gemm::gemv_dispatch(
                    &mut outputs_new,
                    &weights_tensor,
                    &inputs_f32,
                    4, // threads
                    None, None
                );
            }
            let dispatch_elapsed = start.elapsed();
            let dispatch_gflops = flops / dispatch_elapsed.as_secs_f64() / 1e9;
            let dispatch_ms = dispatch_elapsed.as_secs_f64() * 1000.0 / iters as f64;

            println!("  n={}: tiled={:.2}ms ({:.1}GF), dispatch={:.2}ms ({:.1}GF)",
                     n, tiled_ms, tiled_gflops, dispatch_ms, dispatch_gflops);
        }
    }

    println!("\n=== Performance Target ===");
    println!("llama.cpp achieves ~200 GFLOPS for quantized GEMM on M4");
    println!("Memory bandwidth ~100 GB/s, quantized GEMM should be compute-bound");
}
