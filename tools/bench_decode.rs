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

    // Pre-fill with prompt (synchronous)
    let prompt_tokens: Vec<u32> = vec![1];
    let prefill_start = std::time::Instant::now();

    // Multi-pass pipeline: need to call multiple times to fill pipeline and get results
    // Pipeline depth defaults to 2
    let pipeline_depth = std::env::var("INFER_PIPELINE_DEPTH")
        .map(|v| v.parse().unwrap_or(2))
        .unwrap_or(2);

    // Fill pipeline
    for _ in 0..pipeline_depth {
        let _ = gpu_ctx.forward_gpu_batched(&prompt_tokens);
    }

    // Get result (waits for oldest CB in pipeline)
    let logits = gpu_ctx.forward_gpu_batched(&[1]).expect("Prefill failed");

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

    // Decode benchmark with async pipeline
    let num_decode = 10;
    let mut total_time = std::time::Duration::ZERO;

    // Submit first decode (async)
    let mut pending_logits = gpu_ctx.forward_gpu_batched(&[next_token]);

    for i in 0..num_decode {
        let start = std::time::Instant::now();

        // Get previous result and submit next token
        let current_logits = if i < num_decode - 1 {
            // Sample next token first (CPU work overlaps GPU execution)
            let sampled = pending_logits
                .as_ref()
                .map(|l| {
                    l.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i as u32)
                        .unwrap_or(0)
                })
                .unwrap_or(next_token);

            // Submit next token (async) and get previous result
            let result = gpu_ctx.forward_gpu_batched(&[sampled]);
            next_token = sampled;
            let prev_result = pending_logits;
            pending_logits = result;
            prev_result
        } else {
            // Last iteration: get final result
            gpu_ctx.forward_gpu_batched(&[next_token])
        };

        let elapsed = start.elapsed();
        total_time += elapsed;

        // Sample token from result
        if let Some(l) = current_logits.as_ref() {
            next_token = l
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }

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
