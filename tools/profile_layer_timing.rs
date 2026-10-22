// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile single-layer GPU execution time to understand performance breakdown

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
    println!("=== Layer Timing Profile ===\n");

    let mut gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load GPU model");

    println!(
        "Config: hidden={}, heads={}, head_dim={}, layers={}",
        gpu_ctx.config.hidden_size,
        gpu_ctx.config.n_heads,
        gpu_ctx.config.head_dim,
        gpu_ctx.config.n_layers
    );

    // Enable detailed timing
    std::env::set_var("INFER_PROFILE_LAYERS", "1");

    // Warmup
    let _ = gpu_ctx.forward_gpu_batched(&[1]);
    println!("Warmup complete\n");

    // Measure single token generation with layer breakdown
    println!("=== Single Token Decode ===");

    let start = std::time::Instant::now();
    let logits = gpu_ctx.forward_gpu_batched(&[1]).expect("Decode failed");
    let total_time = start.elapsed();

    println!("\nTotal decode time: {:.2} ms", total_time.as_secs_f64() * 1000.0);
    println!("Achieved throughput: {:.2} t/s", 1000.0 / total_time.as_secs_f64() / 1000.0);

    // Check logits
    let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
    println!(
        "Logits: len={}, min={:.4}, max={:.4}, nan={}",
        logits.len(), min, max, nan_count
    );
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This benchmark requires Metal support");
}