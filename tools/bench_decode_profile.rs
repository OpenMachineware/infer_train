// Profile decode timing breakdown
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::{Device, InferenceContext, KVCache, KVCacheFormat};

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

    // Prefill
    let logits = gpu_ctx.forward_gpu_batched(&[1]).expect("Prefill failed");

    // Warmup decode
    let mut next_token: u32 = 0;
    for _ in 0..3 {
        let _ = gpu_ctx.forward_gpu_batched(&[next_token]);
        next_token += 1;
    }

    // Profile decode
    let num_decode = 5;
    println!("\n=== Timing Breakdown ===");

    let mut total_forward = 0u64;
    let mut total_sampling = 0u64;

    for i in 0..num_decode {
        let t0 = std::time::Instant::now();
        let logits = gpu_ctx.forward_gpu_batched(&[i as u32]).expect("Decode failed");
        let t1 = std::time::Instant::now();

        // Sampling (greedy argmax)
        let t2 = std::time::Instant::now();
        let _next_token = logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        let t3 = std::time::Instant::now();

        let forward_us = (t1 - t0).as_micros() as u64;
        let sampling_us = (t3 - t2).as_micros() as u64;

        total_forward += forward_us;
        total_sampling += sampling_us;

        println!("Token {}: forward={:.2}ms, sampling={:.2}ms",
            i,
            forward_us as f64 / 1000.0,
            sampling_us as f64 / 1000.0);
    }

    println!("\nAverage forward: {:.2} ms", total_forward as f64 / num_decode as f64 / 1000.0);
    println!("Average sampling: {:.2} ms", total_sampling as f64 / num_decode as f64 / 1000.0);
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This benchmark requires Metal support");
}
