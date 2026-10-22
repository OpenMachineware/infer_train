// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Benchmark K-quant GEMV optimization: before vs after threadgroup config
//!
//! Usage: cargo run --release --features metal --bin bench-kquant-gemv-optimization -- <model.gguf>

use infer_train::{
    compute::MetalBackend,
    inference::{Device, InferenceContext},
    model::llama::LlamaModel,
};
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    println!("=== K-quant GEMV Threadgroup Optimization Benchmark ===");
    println!("Model: {}", model_path);
    println!();

    // Warm-up run
    println!("Warm-up run...");
    let (tokens, warmup_time) = run_decode(model_path, 5);
    println!("  Warm-up: {} tokens in {} ms", tokens, warmup_time);
    println!();

    // Actual benchmark runs
    let num_runs = 5;
    let tokens_per_run = 10;
    let mut results = Vec::new();

    println!(
        "Running {} iterations ({} tokens each)...",
        num_runs, tokens_per_run
    );
    for i in 0..num_runs {
        let (tokens, time_ms) = run_decode(model_path, tokens_per_run);
        let tps = tokens as f64 / (time_ms as f64 / 1000.0);
        results.push((tokens, time_ms, tps));
        println!(
            "  Run {}: {} tokens in {} ms ({:.2} t/s)",
            i + 1,
            tokens,
            time_ms,
            tps
        );
    }

    // Calculate statistics
    let avg_tps: f64 = results.iter().map(|(_, _, tps)| tps).sum::<f64>()
        / results.len() as f64;
    let min_tps = results
        .iter()
        .map(|(_, _, tps)| tps)
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let max_tps = results
        .iter()
        .map(|(_, _, tps)| tps)
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!();
    println!("=== Results ===");
    println!("Average: {:.2} t/s", avg_tps);
    println!("Min:     {:.2} t/s", min_tps);
    println!("Max:     {:.2} t/s", max_tps);
    println!("Stddev:  {:.2} t/s", calculate_stddev(&results));

    // Also output in format suitable for comparison scripts
    println!();
    println!("RESULT: {:.2} t/s (avg)", avg_tps);
}

fn run_decode(model_path: &str, num_tokens: usize) -> (usize, u128) {
    let device = Device::Gpu;
    let backend = MetalBackend::new().expect("Failed to create Metal backend");
    let gpu_ctx = backend.gpu_ctx.clone();

    let mut model = LlamaModel::from_gguf(model_path, device)
        .expect("Failed to load model");
    let mut ctx = InferenceContext::new(&model, device, 2048)
        .expect("Failed to create context");

    // Pre-upload weights to GPU
    model
        .pre_upload_weights_to_gpu(&gpu_ctx)
        .expect("Failed to pre-upload weights");

    // Prefill with a simple prompt
    let prompt_tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
    let logits =
        ctx.prefill(&model, &prompt_tokens, &gpu_ctx).expect("Prefill failed");

    let start = Instant::now();

    // Generate tokens
    let mut current_logits = logits;
    let mut generated = 0;
    for _ in 0..num_tokens {
        current_logits =
            ctx.forward_step(&model, &gpu_ctx).expect("Forward step failed");
        generated += 1;
    }

    let elapsed = start.elapsed().as_millis();

    (generated, elapsed)
}

fn calculate_stddev(results: &[(usize, u128, f64)]) -> f64 {
    let avg = results.iter().map(|(_, _, tps)| tps).sum::<f64>()
        / results.len() as f64;
    let variance =
        results.iter().map(|(_, _, tps)| (tps - avg).powi(2)).sum::<f64>()
            / results.len() as f64;
    variance.sqrt()
}
