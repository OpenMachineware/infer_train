// Benchmark CPU prefill performance vs llama.cpp
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

    let ctx = InferenceContext::from_gguf(&args[1], Device::Cpu)
        .expect("Failed to load model");

    println!("=== CPU Prefill Benchmark ===");
    println!("Model: {}", args[1]);
    println!(
        "Hidden: {}, Layers: {}",
        ctx.config.hidden_size,
        ctx.model.layers.len()
    );
    println!(
        "Heads: {}/{}, head_dim: {}",
        ctx.config.n_heads, ctx.config.n_kv_heads, ctx.config.head_dim
    );

    // Test prefill with different prompt lengths
    println!("\n--- Prefill (batch processing) ---");
    for prompt_len in [16, 32, 64] {
        let mut kv = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
        let tokens: Vec<u32> = (1..=prompt_len as u32).collect();

        // Benchmark (no warmup for prefill - first run is what matters)
        let start = Instant::now();
        match ctx.model.forward(&tokens, &mut kv) {
            Some(_) => {
                let elapsed = start.elapsed();
                let tps = prompt_len as f64 / elapsed.as_secs_f64();
                println!(
                    "Prefill {} tokens: {:.2}s, {:.2} t/s",
                    prompt_len,
                    elapsed.as_secs_f64(),
                    tps
                );
            }
            None => {
                println!("Prefill {} tokens: FAILED", prompt_len);
            }
        }
    }

    // Test decode for comparison
    println!("\n--- Decode (token-by-token generation) ---");
    let mut kv = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    // Prefill with single token
    let _ = ctx.model.forward(&[1], &mut kv);

    // Decode 20 tokens
    let start = Instant::now();
    let mut successful = 0;
    for _ in 0..20 {
        match ctx.model.forward(&[100], &mut kv) {
            Some(_) => successful += 1,
            None => break,
        }
    }
    let elapsed = start.elapsed();
    let tps = successful as f64 / elapsed.as_secs_f64();
    println!(
        "Decode {} tokens: {:.2}s, {:.2} t/s",
        successful,
        elapsed.as_secs_f64(),
        tps
    );

    println!("\n=== Comparison with llama.cpp ===");
    println!("llama.cpp decode: ~0.1 t/s, prefill: ~0.6 t/s (0.6B model, M4)");
    println!("Our decode: {:.2} t/s", tps);
}
