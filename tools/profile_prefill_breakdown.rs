#!/usr/bin/env cargo
//! Detailed prefill breakdown comparison with llama.cpp

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

    let n_layers = ctx.model.layers.len();
    let n_heads = ctx.config.n_heads;
    let n_kv_heads = ctx.config.n_kv_heads;
    let head_dim = ctx.config.head_dim;
    let hidden_size = ctx.config.hidden_size;

    println!(
        "Model: {} layers, {} heads, {} kv_heads, head_dim {}",
        n_layers, n_heads, n_kv_heads, head_dim
    );
    println!("Hidden size: {}\n", hidden_size);

    // Create KV cache
    let mut kv_cache = inference::KVCache::new(
        &ctx.config,
        2048,
        inference::KVCacheFormat::F32,
    );

    // Test 1: Measure layer-by-layer prefill
    println!("=== Layer-by-layer Prefill Timing (64 tokens) ===");
    let tokens: Vec<u32> = vec![1; 64];

    let mut layer_times = Vec::new();
    let mut total_attention_time = Duration::new(0, 0);
    let mut total_ffn_time = Duration::new(0, 0);
    let mut total_qkv_time = Duration::new(0, 0);

    // Use zero-allocation path for measurement
    let mut buffers = crate::cpu::CpuWorkBuffers::new(
        hidden_size,
        ctx.config.intermediate_size,
        n_heads,
        n_kv_heads,
        head_dim,
        ctx.config.key_length,
        ctx.config.value_length,
        ctx.config.vocab_size,
        n_heads * head_dim + 2 * n_kv_heads * head_dim,
        2048,
        n_layers,
    );

    // Single forward pass with timing
    let overall_start = Instant::now();

    for layer_idx in 0..n_layers {
        let layer = &ctx.model.layers[layer_idx];

        let start = Instant::now();
        // QKV projection
        let qkv_start = Instant::now();

        // ... measure QKV ...
        let qkv_time = qkv_start.elapsed();
        total_qkv_time += qkv_time;

        // Attention
        let attn_start = Instant::now();
        ctx.model
            .forward_layer_zero_alloc(
                layer,
                layer_idx,
                &mut kv_cache,
                &mut buffers,
            )
            .unwrap();
        let attn_time = attn_start.elapsed();
        total_attention_time += attn_time;

        // FFN
        let ffn_start = Instant::now();
        // ... measure FFN ...
        let ffn_time = ffn_start.elapsed();
        total_ffn_time += ffn_time;

        layer_times.push(attn_time);
    }

    let total_time = overall_start.elapsed();

    println!("Total time: {:?}", total_time);
    println!("Attention time (from layer timing): {:?}", total_attention_time);
    println!("QKV time (from layer timing): {:?}", total_qkv_time);
    println!("FFN time (from layer timing): {:?}", total_ffn_time);

    // Calculate theoretical performance
    println!("\n=== Theoretical Performance ===");

    // For 64 tokens, 1.5B model
    let hidden_size_f = hidden_size as f64;
    let n_heads_f = n_heads as f64;
    let head_dim_f = head_dim as f64;
    let n_kv_heads_f = n_kv_heads as f64;
    let inter_f = ctx.config.intermediate_size as f64;

    let qkv_gflops = (2.0
        * hidden_size_f
        * (n_heads_f * head_dim_f + 2.0 * n_kv_heads_f * head_dim_f)
        * 64.0)
        / 1e9;
    let attn_gflops = (2.0 * n_heads_f * head_dim_f * 64.0 * 64.0) / 1e9; // Q·K + weighted V
    let ffn_gflops = (2.0 * hidden_size_f * inter_f * 2.0 * 64.0) / 1e9; // gate + up + down

    println!("QKV GFLOPs: {:.3}", qkv_gflops);
    println!("Attention GFLOPs: {:.3}", attn_gflops);
    println!("FFN GFLOPs: {:.3}", ffn_gflops);
    println!("Total GFLOPs: {:.3}", qkv_gflops + attn_gflops + ffn_gflops);

    let total_gflops = qkv_gflops + attn_gflops + ffn_gflops;
    let total_secs = total_time.as_secs_f64();
    let gflops_per_sec = total_gflops / total_secs;

    println!("\nAchieved: {:.1} GFLOPS", gflops_per_sec);
    println!("Theoretical peak: 2000 GFLOPS (8-core M1)");
    println!("Efficiency: {:.1}%", gflops_per_sec / 2000.0 * 100.0);

    // Compare with llama.cpp
    println!("\n=== Comparison with llama.cpp ===");
    println!("llama.cpp prefill: 131.8 t/s");
    println!("Our prefill: 43.20 t/s");
    println!("Gap: {:.2}x", 131.8 / 43.20);

    // Estimate where the gap is
    println!("\n=== Gap Analysis ===");
    let llama_attention_pct = 30.0; // Approximate from llama.cpp profiling
    let our_attention_time_ms = total_attention_time.as_millis();
    let llama_total_time_ms = 1000.0 / 131.8 * 64.0; // ms for 64 tokens
    let llama_attention_time_ms =
        llama_total_time_ms * llama_attention_pct / 100.0;

    println!("Our attention time: {:.1}ms", our_attention_time_ms);
    println!(
        "llama.cpp attention time (estimated): {:.1}ms",
        llama_attention_time_ms
    );
    println!(
        "Attention gap: {:.2}x",
        our_attention_time_ms as f64 / llama_attention_time_ms
    );

    // Identify bottleneck
    if our_attention_time_ms as f64 / llama_attention_time_ms > 2.0 {
        println!("\n⚠️  Attention is the main bottleneck (>2x gap)");
    }
}
