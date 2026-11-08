#!/usr/bin/env cargo
//! Debug tool to find where prefill hangs

use infer_train::inference::KVCacheBase;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return;
    }

    let model_path = &args[1];
    println!("Loading model from: {}", model_path);

    // Load model
    let ctx = match infer_train::inference::InferenceContext::from_gguf(
        model_path,
        infer_train::inference::Device::Cpu,
    ) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    };

    println!("Model loaded successfully");
    println!(
        "Config: {} layers, {} heads, {} kv_heads, head_dim {}",
        ctx.model.layers.len(),
        ctx.config.n_heads,
        ctx.config.n_kv_heads,
        ctx.config.head_dim
    );

    // Create KV cache
    let mut kv_cache = infer_train::inference::KVCache::new(
        &ctx.config,
        2048,
        infer_train::inference::KVCacheFormat::F32,
    );

    // Create buffer pool and get buffers (this properly initializes norm_weights)
    let mut buffer_pool = infer_train::cpu::CpuWorkBuffersPool::new(&ctx.model);
    buffer_pool.preallocate(1);
    let mut buffers =
        buffer_pool.allocate().expect("Failed to allocate buffers");

    // Test single token decode first
    println!("\n=== Testing single token decode ===");
    println!("Model uses MQA: {}", ctx.model.uses_mqa());
    println!("Is decode (tokens.len() == 1): true");
    kv_cache.clear();

    let tokens: Vec<u32> = vec![1];
    println!("Calling forward_pooled...");
    let start = Instant::now();
    match ctx.model.forward_pooled(&tokens, &mut kv_cache, &mut buffers) {
        Some(logits) => {
            println!("✅ Decode succeeded in {:?}", start.elapsed());
            println!("   Logits len: {}", logits.len());
        }
        None => {
            println!("❌ Decode failed");
        }
    }

    // Test prefill with increasing token counts
    for n_tokens in [2, 4, 8] {
        println!("\n=== Testing {}-token prefill ===", n_tokens);
        kv_cache.clear();

        let tokens: Vec<u32> = vec![1; n_tokens];
        let start = Instant::now();

        println!("Starting forward...");
        match ctx.model.forward_pooled(&tokens, &mut kv_cache, &mut buffers) {
            Some(logits) => {
                println!("✅ Prefill succeeded in {:?}", start.elapsed());
                println!("   Logits len: {}", logits.len());
            }
            None => {
                println!("❌ Prefill failed");
            }
        }

        // Timeout check
        if start.elapsed().as_secs() > 5 {
            println!(
                "⚠️ Prefill appears to be hanging after {:?}",
                start.elapsed()
            );
            break;
        }
    }
}
