// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test full GPU forward pass to find which layer first produces NaN

use std::env;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let args: Vec<String> = env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: {} <model.gguf>", args[0]);
            std::process::exit(1);
        }

        let model_path = &args[1];
        println!("Model: {}", model_path);

        let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load GPU model");

        // Test with single token
        let tokens = vec![1u32];

        println!("\n=== Running full GPU forward pass ===");
        let result = ctx.forward_gpu_native(&tokens);

        match result {
            Some(logits) => {
                let nan_count = logits.iter().filter(|x| x.is_nan()).count();
                println!(
                    "Final logits: {} NaN out of {}",
                    nan_count,
                    logits.len()
                );

                if nan_count > 0 {
                    println!("\n!!! NaN detected in final output !!!");

                    // Check if it's 100% NaN
                    if nan_count == logits.len() {
                        println!("100% NaN - something is very wrong");

                        // Try CPU to verify model is correct
                        println!("\nTrying CPU inference to verify model...");
                        let cpu_ctx = InferenceContext::from_gguf(
                            model_path,
                            Device::Cpu,
                        )
                        .expect("Failed to load CPU model");
                        let mut cpu_kv = infer_train::inference::KVCache::new(
                            &cpu_ctx.config,
                            2048,
                            infer_train::inference::KVCacheFormat::F32,
                        );
                        let cpu_logits = cpu_ctx
                            .model
                            .forward(&tokens, &mut cpu_kv)
                            .expect("CPU forward failed");
                        let cpu_nan =
                            cpu_logits.iter().filter(|x| x.is_nan()).count();
                        println!(
                            "CPU inference: {} NaN out of {}",
                            cpu_nan,
                            cpu_logits.len()
                        );
                    }
                } else {
                    let valid: Vec<f32> = logits
                        .iter()
                        .filter(|x| x.is_finite())
                        .cloned()
                        .collect();
                    println!(
                        "Valid range: [{:.6}, {:.6}]",
                        valid.iter().cloned().fold(f32::INFINITY, f32::min),
                        valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                    );
                }
            }
            None => {
                println!(
                    "GPU inference returned None - trying second call for async pipeline"
                );
                let result2 = ctx.forward_gpu_native(&tokens);
                match result2 {
                    Some(logits) => {
                        let nan_count =
                            logits.iter().filter(|x| x.is_nan()).count();
                        println!(
                            "Second call: {} NaN out of {}",
                            nan_count,
                            logits.len()
                        );
                    }
                    None => {
                        println!("Second call also returned None");
                    }
                }
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}
