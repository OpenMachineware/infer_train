#!/usr/bin/env cargo
//! 测量实际forward pass各部分绝对时间

use infer_train::*;
use std::time::{Duration, Instant};

fn main() {
    // 加载模型
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return;
    }

    let mut ctx = inference::InferenceContext::from_gguf(&args[1], inference::Device::Cpu)
        .expect("Failed to load model");

    let n_layers = ctx.model.layers.len();
    let n_heads = ctx.config.n_heads;
    let n_kv_heads = ctx.config.n_kv_heads;
    let head_dim = ctx.config.head_dim;
    let hidden_size = ctx.config.hidden_size;

    println!("=== 实际Forward Pass绝对时间测量 ===");
    println!("模型: {} layers, {} heads, hidden_size={}\n",
             n_layers, n_heads, hidden_size);

    // 创建KV cache
    let mut kv_cache = inference::KVCache::new(
        &ctx.config,
        2048,
        inference::KVCacheFormat::F32,
    );

    // 测试prefill (64 tokens)
    let tokens: Vec<u32> = vec![1; 64];

    let mut total_forward_time = Duration::new(0, 0);
    let mut total_embedding_time = Duration::new(0, 0);
    let mut total_layer_time = vec![Duration::new(0, 0); n_layers];

    let iterations = 10;

    for iter in 0..iterations {
        kv_cache.clear();

        // Embedding
        let emb_start = Instant::now();
        let embedded = ctx.model.embed_tokens(&tokens).expect("embedding failed");
        total_embedding_time += emb_start.elapsed();

        // Forward pass
        let forward_start = Instant::now();

        // QKV + Attention + FFN per layer
        for layer_idx in 0..n_layers {
            let layer_start = Instant::now();
            ctx.model.forward_layer(layer_idx, &mut kv_cache, None).expect("forward failed");
            total_layer_time[layer_idx] += layer_start.elapsed();
        }

        total_forward_time += forward_start.elapsed();
    }

    println!("=== 绝对时间（{}次平均） ===", iterations);
    println!("Embedding: {:?}", total_embedding_time / iterations);

    for layer_idx in 0..n_layers.min(5) {
        println!("Layer {}: {:?}", layer_idx, total_layer_time[layer_idx] / iterations);
    }
    if n_layers > 5 {
        let remaining: Duration = total_layer_time[5..].iter().sum();
        println!("Layers 5-{}: {:?}", n_layers - 1, remaining / iterations);
    }

    println!("\n总forward时间: {:?}", total_forward_time / iterations);

    let tokens_per_sec = 64.0 / (total_forward_time / iterations).as_secs_f64();
    println!("吞吐量: {:.1} t/s", tokens_per_sec);
}
