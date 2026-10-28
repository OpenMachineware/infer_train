// Simple GEMV benchmark to measure single GEMV performance

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use infer_train::model::tensor::Tensor;
use infer_train::quant::kv_cache::quantize_row_q8_k_neon;
use infer_train::quant::matmul::k_quant::matmul_q4_k_q8_k;
use infer_train::quant::types::QK_K;
use infer_train::quant::types::{BlockQ4K, BlockQ8K};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== GEMV Performance Benchmark ===\n");
    println!("Model: {}", model_path);

    // Load model
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    println!("Model loaded successfully\n");

    // Get layer 0 weights
    let layer = &ctx.model.layers[0];
    let hidden_size = ctx.config.hidden_size;

    // Test sizes
    let sizes = [
        ("Q projection", layer.wq.as_ref().unwrap()),
        ("Gate projection", layer.w1.as_ref().unwrap()),
        ("Up projection", layer.w3.as_ref().unwrap()),
        ("Down projection", layer.w2.as_ref().unwrap()),
        ("Output projection", ctx.model.output.as_ref().unwrap()),
    ];

    // Create input vector
    let x: Vec<f32> = vec![0.5; hidden_size];

    // Quantize input once
    let nb = (hidden_size + QK_K - 1) / QK_K;
    let mut x_q8k = vec![unsafe { std::mem::zeroed() }; nb];
    quantize_row_q8_k_neon(&x, &mut x_q8k);

    for (name, weight) in &sizes {
        let out_size = if weight.shape[0] == hidden_size {
            weight.shape[1] // Normal: [K, M]
        } else {
            weight.shape[0] // Transposed: [M, K]
        };

        let dtype = weight.dtype;
        println!("{}: {}x{} ({:?})", name, hidden_size, out_size, dtype);

        // Warmup
        for _ in 0..5 {
            let _ = ctx.model.gemv(&x, weight, out_size);
        }

        // Benchmark
        let n_runs = 10;
        let start = Instant::now();
        for _ in 0..n_runs {
            let _ = ctx.model.gemv(&x, weight, out_size);
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_millis() as f64 / n_runs as f64;
        let flops = 2.0 * hidden_size as f64 * out_size as f64;
        let gflops = flops / (avg_ms / 1000.0) / 1e9;
        println!("  Time: {:.3} ms, GFLOPS: {:.1}", avg_ms, gflops);
    }
}
