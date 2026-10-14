// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test GPU inference end-to-end

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model: {}", model_path);

    // Try GPU first
    println!("\n=== GPU Inference ===");
    let gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu);

    match gpu_ctx {
        Ok(mut ctx) => {
            println!("✓ Model loaded on GPU");
            println!("  Layers: {}", ctx.layer_count());
            println!("  Params: {}M", ctx.param_count() / 1_000_000);

            let tokens: Vec<u32> = vec![1, 2, 3];
            println!("\nInput tokens: {:?}", tokens);

            let start = std::time::Instant::now();
            match ctx.forward_gpu_native(&tokens) {
                Some(logits) => {
                    let elapsed = start.elapsed();
                    let non_zero = logits.iter().filter(|&&v| v != 0.0).count();
                    let num_positive =
                        logits.iter().filter(|&&v| v > 0.0).count();
                    let num_negative =
                        logits.iter().filter(|&&v| v < 0.0).count();
                    let num_nan =
                        logits.iter().filter(|&&v| v.is_nan()).count();
                    let num_inf =
                        logits.iter().filter(|&&v| v.is_infinite()).count();
                    let max_idx = logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i);
                    let max_val = logits
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let min_val =
                        logits.iter().cloned().fold(f32::INFINITY, f32::min);

                    println!(
                        "  Logits: {} values, {} non-zero",
                        logits.len(),
                        non_zero
                    );
                    println!(
                        "  Stats: positive={}, negative={}, nan={}, inf={}",
                        num_positive, num_negative, num_nan, num_inf
                    );
                    println!("  Range: [{}, {}]", min_val, max_val);
                    println!("  Max logit: {} at index {:?}", max_val, max_idx);
                    println!("  Time: {:?}", elapsed);

                    // Show first 10 values
                    println!("  First 10: {:?}", &logits[..10]);
                }
                None => {
                    println!("  ERROR: GPU forward pass failed");
                }
            }
        }
        Err(e) => {
            println!("✗ Failed to load model on GPU: {}", e);
        }
    }

    // Compare with CPU
    println!("\n=== CPU Inference (for comparison) ===");
    let mut cpu_ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model on CPU");

    println!("✓ Model loaded on CPU");

    let config = cpu_ctx.config.clone();
    let mut kv_cache = KVCache::new(&config, 10, KVCacheFormat::F32);

    let tokens: Vec<u32> = vec![1, 2, 3];
    println!("Input tokens: {:?}", tokens);

    let start = std::time::Instant::now();
    match cpu_ctx.model.forward(&tokens, &mut kv_cache) {
        Some(logits) => {
            let elapsed = start.elapsed();
            let non_zero = logits.iter().filter(|&&v| v != 0.0).count();
            let num_positive = logits.iter().filter(|&&v| v > 0.0).count();
            let num_negative = logits.iter().filter(|&&v| v < 0.0).count();
            let num_nan = logits.iter().filter(|&&v| v.is_nan()).count();
            let num_inf = logits.iter().filter(|&&v| v.is_infinite()).count();
            let max_idx = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);
            let max_val =
                logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_val = logits.iter().cloned().fold(f32::INFINITY, f32::min);

            println!(
                "  Logits: {} values, {} non-zero",
                logits.len(),
                non_zero
            );
            println!(
                "  Stats: positive={}, negative={}, nan={}, inf={}",
                num_positive, num_negative, num_nan, num_inf
            );
            println!("  Range: [{}, {}]", min_val, max_val);
            println!("  Max logit: {} at index {:?}", max_val, max_idx);
            println!("  Time: {:?}", elapsed);

            // Show first 10 values
            println!("  First 10: {:?}", &logits[..10]);
        }
        None => {
            println!("  ERROR: CPU forward pass failed");
        }
    }

    println!("\n=== Done ===");
}
