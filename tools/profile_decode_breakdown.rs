#!/usr/bin/env cargo
//! Profile decode to find bottlenecks - simplified version using forward_pooled

use infer_train::inference::KVCacheBase;
use infer_train::*;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return;
    }

    let model_path = &args[1];

    // Load model
    let mut ctx = inference::InferenceContext::from_gguf(
        model_path,
        inference::Device::Cpu,
    )
    .expect("Failed to load model");

    let hidden_size = ctx.config.hidden_size;
    let n_layers = ctx.model.layers.len();
    let n_heads = ctx.config.n_heads;
    let n_kv_heads = ctx.config.n_kv_heads;
    let head_dim = ctx.config.head_dim;

    println!(
        "Model: {} layers, {} heads, head_dim {}",
        n_layers, n_heads, head_dim
    );

    // Create KV cache
    let mut kv_cache = inference::KVCache::new(
        &ctx.config,
        2048,
        inference::KVCacheFormat::F32,
    );

    // Prefill first token to set up KV cache (same as profile-cpu-forward)
    let start = Instant::now();
    ctx.forward_pooled(&[1], &mut kv_cache);
    println!("Prefill 1 token: {:?}", start.elapsed());

    // Profile decode (single token at a time)
    let n_decode = 20;
    let mut total_time = Duration::new(0, 0);

    let start = Instant::now();
    for _ in 0..n_decode {
        let _ = ctx.forward_pooled(&[100], &mut kv_cache);
    }
    total_time = start.elapsed();

    let tps = n_decode as f64 / total_time.as_secs_f64();
    println!(
        "\nDecode {} tokens: {:.2}s, {:.2} t/s",
        n_decode,
        total_time.as_secs_f64(),
        tps
    );

    // Now profile with llama.cpp baseline
    println!("\n=== Comparing with llama.cpp baseline ===");
    println!("Baseline from commit 575e4f6: 39 t/s decode");
    println!("Current: {:.2} t/s decode", tps);
    println!("Gap: {:.2}x", 39.0 / tps);
}
