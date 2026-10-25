// Test async pipeline correctness
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::inference::Device;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Model: {}", model_path);

    let mut ctx = infer_train::inference::InferenceContext::from_gguf(
        model_path,
        Device::Gpu,
    )
    .expect("Failed to load GPU model");

    println!(
        "Config: hidden={}, heads={}, head_dim={}",
        ctx.config.hidden_size,
        ctx.config.n_heads,
        ctx.config.head_dim
    );

    // Test sync vs async decode
    println!("\n=== Test 1: Synchronous decode ===");
    let tokens = vec![1u32];

    let start = Instant::now();
    let result1 = ctx.forward_gpu_batched(&tokens);
    let time1 = start.elapsed();

    println!("First call: {:?}", time1);
    match result1 {
        Some(logits) => {
            let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
            println!("  Logits: {} values, {} NaN", logits.len(), nan_count);
            println!("  Expected: First call should wait synchronously and return logits");
        }
        None => {
            println!("  Result: None (this is expected for async pipeline first call)");
        }
    }

    let start = Instant::now();
    let result2 = ctx.forward_gpu_batched(&tokens);
    let time2 = start.elapsed();

    println!("Second call: {:?}", time2);
    match result2 {
        Some(logits) => {
            let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
            println!("  Logits: {} values, {} NaN", logits.len(), nan_count);
            println!("  Expected: Second call should return first call's logits");
        }
        None => {
            println!("  ERROR: Second call should return logits!");
        }
    }

    // Test multiple decode steps
    println!("\n=== Test 2: Multiple decode steps ===");
    let mut total_time = Duration::ZERO;

    for i in 0..5 {
        let start = Instant::now();
        let result = ctx.forward_gpu_batched(&[1]);
        let elapsed = start.elapsed();
        total_time += elapsed;

        if let Some(logits) = result {
            let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
            println!(
                "  Step {}: {:?}, logits_len={}, nan={}",
                i,
                elapsed,
                logits.len(),
                nan_count
            );
        } else {
            println!("  Step {}: None (should not happen after first call)", i);
        }
    }

    let avg_time = total_time.as_millis() as f64 / 5.0;
    let tps = 1000.0 / avg_time;
    println!("  Average: {:.2} ms/token ({:.2} t/s)", avg_time, tps);
}
