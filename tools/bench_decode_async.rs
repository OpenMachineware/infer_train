// Benchmark decode with async pipeline (measuring overlap potential)
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::{Device, InferenceContext};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Model: {}", model_path);

    let mut gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load GPU model");

    println!(
        "Config: hidden={}, heads={}, head_dim={}",
        gpu_ctx.config.hidden_size,
        gpu_ctx.config.n_heads,
        gpu_ctx.config.head_dim
    );

    // Prefill (synchronous)
    let _ = gpu_ctx.forward_gpu_batched(&[1]);
    let logits = gpu_ctx.forward_gpu_batched(&[1]).expect("Prefill failed");

    let mut next_token = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0);

    println!("\n=== Async Pipeline Test ===\n");

    // Test 1: Sync baseline
    println!("Test 1: Synchronous (baseline)");
    let num_decode = 10;
    let mut total_time = std::time::Duration::ZERO;

    for _ in 0..num_decode {
        let start = std::time::Instant::now();
        let _ = gpu_ctx.forward_gpu_batched(&[next_token]);
        let elapsed = start.elapsed();
        total_time += elapsed;
    }
    println!(
        "  Average: {:.2} ms/token",
        total_time.as_millis() as f64 / num_decode as f64
    );

    // Test 2: Async with simulated CPU work
    println!("\nTest 2: Async with 5ms CPU work (simulating sampling)");

    let mut total_time = std::time::Duration::ZERO;
    let mut pending = gpu_ctx.forward_gpu_batched(&[next_token]);

    for _ in 0..num_decode {
        let start = std::time::Instant::now();

        // Simulate CPU work (sampling, tokenization, etc.)
        std::thread::sleep(std::time::Duration::from_millis(5));

        // Get result from previous call and submit next
        let result = gpu_ctx.forward_gpu_batched(&[next_token]);
        pending = result;

        let elapsed = start.elapsed();
        total_time += elapsed;
    }

    println!(
        "  Average: {:.2} ms/call (including 5ms CPU work)",
        total_time.as_millis() as f64 / num_decode as f64
    );

    // Test 3: Pure async (no CPU work)
    println!("\nTest 3: Pure async (no CPU work)");

    let mut total_time = std::time::Duration::ZERO;
    let mut pending = gpu_ctx.forward_gpu_batched(&[next_token]);

    for _ in 0..num_decode {
        let start = std::time::Instant::now();

        let result = gpu_ctx.forward_gpu_batched(&[next_token]);
        pending = result;

        let elapsed = start.elapsed();
        total_time += elapsed;
    }

    println!(
        "  Average: {:.2} ms/call",
        total_time.as_millis() as f64 / num_decode as f64
    );
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This benchmark requires Metal support");
}