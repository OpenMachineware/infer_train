// Profile encoding vs execution time
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

    // Warm up
    let _ = gpu_ctx.forward_gpu_batched(&[1]);

    // Measure multiple forward passes
    let num_iterations = 20;
    let mut encoding_times = Vec::new();
    let mut total_times = Vec::new();

    for i in 0..num_iterations {
        let total_start = std::time::Instant::now();
        let _ = gpu_ctx.forward_gpu_batched(&[1]);
        let total_time = total_start.elapsed();

        total_times.push(total_time.as_nanos() as f64 / 1_000_000.0);
    }

    println!("\n=== Timing Results ===");
    let avg_total: f64 = total_times.iter().sum::<f64>() / num_iterations as f64;
    println!(
        "Average total time: {:.2} ms ({:.1} t/s)",
        avg_total,
        1000.0 / avg_total
    );

    // Print individual times
    println!("\nIteration times (ms):");
    for (i, t) in total_times.iter().enumerate() {
        println!("  {}: {:.2}", i, t);
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!(
        "This benchmark requires Metal support (macOS with metal feature)"
    );
}