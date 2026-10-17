// Benchmark decode performance (single token generation)
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Model: {}", model_path);

    // GPU decode benchmark with batched encoding
    println!("\n=== GPU Decode Benchmark (Batched) ===");
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
    let logits =
        gpu_ctx.forward_gpu_batched(&prompt_tokens).expect("Prefill failed");
    let prefill_time = prefill_start.elapsed();

    // Debug: check logits
    let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().sum();
    let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
    let zero_count = logits.iter().filter(|&&x| x == 0.0).count();
    println!(
        "Logits: len={}, min={:.4}, max={:.4}, mean={:.4}, nan={}, zero={}",
        logits.len(),
        min,
        max,
        sum / logits.len() as f32,
        nan_count,
        zero_count
    );

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
    let mut logits = logits;

    for i in 0..num_decode {
        let start = std::time::Instant::now();
        logits =
            gpu_ctx.forward_gpu_batched(&[next_token]).expect("Decode failed");
        let elapsed = start.elapsed();
        total_time += elapsed;

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
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!(
        "This benchmark requires Metal support (macOS with metal feature)"
    );
}
