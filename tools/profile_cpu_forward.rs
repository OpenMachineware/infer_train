// Profile CPU forward pass to identify bottlenecks
// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let mut ctx = InferenceContext::from_gguf(&args[1], Device::Cpu)
        .expect("Failed to load model");

    let hidden_size = ctx.config.hidden_size;
    let n_layers = ctx.model.layers.len();
    let n_heads = ctx.config.n_heads;
    let n_kv_heads = ctx.config.n_kv_heads;
    let head_dim = ctx.config.head_dim;

    println!("=== CPU Forward Profile ===");
    println!("Model: {}", args[1]);
    println!("Hidden: {}, Layers: {}", hidden_size, n_layers);
    println!("Heads: {}/{}, head_dim: {}", n_heads, n_kv_heads, head_dim);

    // Profile single layer components
    println!("\n--- Profiling layer 0 components ---");

    let mut kv = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let tokens: Vec<u32> = (1..=16).collect();

    // Measure embedding lookup
    let start = Instant::now();
    for _ in 0..100 {
        for &token in &tokens {
            let _ = ctx.model.embedding_lookup(token);
        }
    }
    let embed_time = start.elapsed();
    println!(
        "Embedding lookup (16 tokens x 100 iters): {:.2}ms total, {:.2}µs per lookup",
        embed_time.as_secs_f64() * 1000.0,
        embed_time.as_secs_f64() * 1_000_000.0 / (16.0 * 100.0)
    );

    // Measure full prefill
    println!("\n--- Full prefill timing ---");
    for n_tokens in [16, 32, 64] {
        let mut kv = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
        let tokens: Vec<u32> = (1..=n_tokens as u32).collect();

        let start = Instant::now();
        let _ = ctx.forward_pooled(&tokens, &mut kv);
        let elapsed = start.elapsed();
        let tps = n_tokens as f64 / elapsed.as_secs_f64();
        println!(
            "Prefill {} tokens: {:.2}s, {:.2} t/s",
            n_tokens,
            elapsed.as_secs_f64(),
            tps
        );
    }

    // Measure decode
    println!("\n--- Decode timing ---");
    let mut kv = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
    let _ = ctx.forward_pooled(&[1], &mut kv);

    let start = Instant::now();
    for _ in 0..20 {
        let _ = ctx.forward_pooled(&[100], &mut kv);
    }
    let elapsed = start.elapsed();
    let tps = 20.0 / elapsed.as_secs_f64();
    println!("Decode 20 tokens: {:.2}s, {:.2} t/s", elapsed.as_secs_f64(), tps);

    // Estimate FLOPs
    println!("\n=== FLOPs estimation ===");
    let qkv_size = n_heads * head_dim + 2 * n_kv_heads * head_dim;
    let intermediate_size = 8960; // Approximate for 0.6B model
    let vocab_size = ctx.config.vocab_size;

    let flops_qkv = 2 * hidden_size * qkv_size;
    let flops_wo = 2 * n_heads * head_dim * hidden_size;
    let flops_ffn = 2 * hidden_size * intermediate_size * 2; // gate + up
    let flops_w2 = 2 * intermediate_size * hidden_size;
    let flops_layer = flops_qkv + flops_wo + flops_ffn + flops_w2;
    let total_flops = n_layers * flops_layer;

    println!("Per layer FLOPs:");
    println!("  QKV: {:.2} M", flops_qkv as f64 / 1e6);
    println!("  wo: {:.2} M", flops_wo as f64 / 1e6);
    println!("  FFN: {:.2} M", flops_ffn as f64 / 1e6);
    println!("  w2: {:.2} M", flops_w2 as f64 / 1e6);
    println!("  Total per layer: {:.2} M", flops_layer as f64 / 1e6);
    println!(
        "  Total for {} layers: {:.2} M",
        n_layers,
        total_flops as f64 / 1e6
    );

    // Calculate GFLOPS
    let best_tps = 63.0; // From benchmark
    let gflops = (total_flops as f64 * best_tps) / 1e9;
    println!(
        "\nAchieved: {:.2} GFLOPS ({:.2} t/s with {:.2} M FLOPs per token)",
        gflops,
        best_tps,
        total_flops as f64 / 1e6
    );

    // Compare with theoretical peak
    // M4 has ~2 TFLOPS FP32 theoretical peak
    let theoretical_peak = 2000.0; // GFLOPS
    let efficiency = gflops / theoretical_peak * 100.0;
    println!(
        "Theoretical peak: {:.2} GFLOPS, efficiency: {:.2}%",
        theoretical_peak, efficiency
    );
}
