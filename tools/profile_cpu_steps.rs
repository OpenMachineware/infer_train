// Profile CPU inference by step

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("=== CPU Inference Profile ===\n");
    println!("Model: {}", model_path);

    // Load model
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");
    println!(
        "Model loaded: {} layers, hidden={}",
        ctx.model.layers.len(),
        ctx.model.config.hidden_size
    );
    println!(
        "n_heads={}, head_dim={}, n_kv_heads={}",
        ctx.model.config.n_heads,
        ctx.model.config.head_dim,
        ctx.model.config.n_kv_heads
    );

    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Warmup
    println!("\n=== Warmup (3 tokens) ===");
    let tokens: Vec<u32> = vec![1, 2, 3];
    let _ = ctx.model.forward(&tokens, &mut kv_cache);
    kv_cache.clear();
    println!("Warmup complete\n");

    // Profile single token forward
    println!("=== Profile Single Token Forward ===");

    let mut total_times = Vec::new();

    for run in 0..10 {
        kv_cache.clear();

        let t0 = Instant::now();
        let _ = ctx.model.forward(&[1u32], &mut kv_cache);
        let elapsed = t0.elapsed();
        total_times.push(elapsed.as_nanos());

        if run == 0 {
            println!("Run {}: {:.2} ms", run, elapsed.as_secs_f64() * 1000.0);
        }
    }

    let avg_ns: u128 = total_times.iter().sum::<u128>() / 10;
    let min_ns = *total_times.iter().min().unwrap();
    let max_ns = *total_times.iter().max().unwrap();

    println!("\n=== Results (10 runs) ===");
    println!("Average: {:.2} ms", avg_ns as f64 / 1_000_000.0);
    println!("Min:     {:.2} ms", min_ns as f64 / 1_000_000.0);
    println!("Max:     {:.2} ms", max_ns as f64 / 1_000_000.0);

    // Estimate per-component time based on model size
    let n_layers = ctx.model.layers.len();
    let hidden = ctx.model.config.hidden_size;

    println!("\n=== Estimated per-layer time ===");
    let per_layer_ns = avg_ns as f64 / n_layers as f64;
    println!("Per layer: {:.2} ms", per_layer_ns / 1_000_000.0);

    // Each layer has:
    // - 1 RMS norm (input)
    // - 3 GEMV (Q, K, V projections) or 1 fused GEMV
    // - 1 attention
    // - 1 GEMV (output projection)
    // - 1 residual add
    // - 1 RMS norm (FFN)
    // - 3 GEMV (gate, up, down) or 2 GEMV (gate+up, down)
    // - 1 SwiGLU activation
    // - 1 residual add

    println!("\n=== Expected operations per layer ===");
    println!("- 2 RMS norms");
    println!("- 3-4 GEMV (QKV + output) or 4-6 GEMV if separate Q,K,V");
    println!("- 2-3 GEMV for FFN (gate, up, down)");
    println!("- 1 attention (decode: O(n) with KV cache)");
    println!("- 2 residual adds");

    println!("\n=== Compare with llama.cpp ===");
    println!(
        "Run: ./llama-cli -m {} -p \"test\" -n 1 -t 10 --verbose",
        model_path
    );
}
