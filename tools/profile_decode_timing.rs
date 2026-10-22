// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile actual decode execution time with GPU timing

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::{Device, InferenceContext};

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let model_path = &args[1];
        println!("=== Decode Time Breakdown ===\n");

        let mut gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load model");

        println!("Config: hidden={}, layers={}\n",
            gpu_ctx.config.hidden_size,
            gpu_ctx.config.n_layers
        );

        // Warmup
        let _ = gpu_ctx.forward_gpu_batched(&[1]);
        println!("Warmup complete\n");

        // Test multiple tokens
        let mut tokens = vec![1];
        let num_tokens = 10;

        println!("Decoding {} tokens...\n", num_tokens);

        for i in 0..num_tokens {
            let start = Instant::now();
            let logits = gpu_ctx.forward_gpu_batched(&tokens).expect("Forward failed");
            let elapsed = start.elapsed();

            let next_token = logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

            println!("  Token {}: {} ({:.2}ms)", i+1, next_token, elapsed.as_secs_f64() * 1000.0);
            tokens.push(next_token);
        }

        println!("\nNote: This measures wall-clock time including:");
        println!("  - CPU-side preparation");
        println!("  - GPU execution");
        println!("  - CPU-side post-processing");
        println!("\nThe gap vs llama.cpp could be in any of these.");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
