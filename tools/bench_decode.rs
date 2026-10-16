// Benchmark decode performance (single token generation)
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Model: {}", model_path);

    // GPU decode benchmark
    println!("\n=== GPU Decode Benchmark ===");
    let mut gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load GPU model");
    println!(
        "Config: hidden={}, heads={}, head_dim={}",
        gpu_ctx.config.hidden_size,
        gpu_ctx.config.n_heads,
        gpu_ctx.config.head_dim
    );

    // Pre-fill with prompt
    let prompt_tokens: Vec<u32> = vec![1];
    let prefill_start = std::time::Instant::now();
    let mut logits =
        gpu_ctx.forward_gpu_zerocopy(&prompt_tokens).expect("Prefill failed");
    let prefill_time = prefill_start.elapsed();
    println!("Prefill: {} ms", prefill_time.as_millis());

    // Sample first token from prefill output
    let mut next_token = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    println!("Generated: {}", next_token);

    // Decode benchmark
    let num_decode = 10;
    let mut total_time = std::time::Duration::ZERO;

    for i in 0..num_decode {
        let start = std::time::Instant::now();
        // Pass ONLY the newly generated token
        logits =
            gpu_ctx.forward_gpu_zerocopy(&[next_token]).expect("Decode failed");
        let elapsed = start.elapsed();
        total_time += elapsed;

        // Sample next token
        next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        println!(
            "  Token {}: {} ({} ms)",
            i + 1,
            next_token,
            elapsed.as_millis()
        );
    }

    let avg_time = total_time.as_millis() as f64 / num_decode as f64;
    let tps = 1000.0 / avg_time;
    println!("\n  Average: {:.2} ms/token ({:.2} t/s)", avg_time, tps);

    // CPU decode benchmark
    println!("\n=== CPU Decode Benchmark ===");
    let cpu_ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load CPU model");
    let config = cpu_ctx.config.clone();
    let mut kv_cache = KVCache::new(&config, 100, KVCacheFormat::F32);

    // Pre-fill
    let prefill_start = std::time::Instant::now();
    let mut logits = cpu_ctx
        .model
        .forward(&prompt_tokens, &mut kv_cache)
        .expect("Prefill failed");
    let prefill_time = prefill_start.elapsed();
    println!("Prefill: {} ms", prefill_time.as_millis());

    // Sample first token
    let mut next_token = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0);
    println!("Generated: {}", next_token);

    // Decode benchmark
    let mut total_time = std::time::Duration::ZERO;

    for i in 0..num_decode {
        let start = std::time::Instant::now();
        // Pass ONLY the newly generated token
        logits = cpu_ctx
            .model
            .forward(&[next_token], &mut kv_cache)
            .expect("Decode failed");
        let elapsed = start.elapsed();
        total_time += elapsed;

        // Sample next token
        next_token = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        println!(
            "  Token {}: {} ({} ms)",
            i + 1,
            next_token,
            elapsed.as_millis()
        );
    }

    let avg_time = total_time.as_millis() as f64 / num_decode as f64;
    let tps = 1000.0 / avg_time;
    println!("\n  Average: {:.2} ms/token ({:.2} t/s)", avg_time, tps);

    println!("\n=== Baseline (llama.cpp) ===");
    println!("  Target: 167 t/s (from previous benchmark)");
}
