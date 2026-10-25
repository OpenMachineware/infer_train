// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Trace NaN layer by layer

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
    println!("Model: {}", model_path);

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load GPU model");

    // Check output layer
    if let Some(output) = &ctx.model.output {
        println!("\nOutput layer:");
        println!("  dtype: {:?}", output.dtype);
        println!("  shape: {:?}", output.shape);
    }

    // Test with CPU first to establish baseline
    println!("\n=== Testing CPU inference ===");
    let tokens: Vec<u32> = (1..=5).collect();
    let cpu_ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load CPU model");

    let mut cpu_kv = KVCache::new(&cpu_ctx.config, 2048, KVCacheFormat::F32);

    let cpu_logits = cpu_ctx
        .model
        .forward(&tokens, &mut cpu_kv)
        .expect("CPU forward failed");
    let cpu_nan = cpu_logits.iter().filter(|x| x.is_nan()).count();
    println!("CPU logits: total={}, NaN={}", cpu_logits.len(), cpu_nan);

    if cpu_nan < cpu_logits.len() / 2 {
        let valid: Vec<f32> = cpu_logits
            .iter()
            .filter(|x| x.is_finite() && **x != 0.0)
            .cloned()
            .collect();
        if !valid.is_empty() {
            println!(
                "CPU valid range: [{:.6}, {:.6}]",
                valid.iter().cloned().fold(f32::INFINITY, f32::min),
                valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }
    }

    // Now test GPU
    println!("\n=== Testing GPU inference ===");
    let gpu_logits = ctx.forward_gpu_native(&tokens);
    match gpu_logits {
        Some(l) => {
            let gpu_nan = l.iter().filter(|x| x.is_nan()).count();
            println!("GPU logits: total={}, NaN={}", l.len(), gpu_nan);

            if gpu_nan < l.len() / 2 {
                let valid: Vec<f32> = l
                    .iter()
                    .filter(|x| x.is_finite() && **x != 0.0)
                    .cloned()
                    .collect();
                if !valid.is_empty() {
                    println!(
                        "GPU valid range: [{:.6}, {:.6}]",
                        valid.iter().cloned().fold(f32::INFINITY, f32::min),
                        valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                    );
                }
            }
        }
        None => println!("GPU inference failed"),
    }
}
