// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug GPU forward step by step to find NaN source

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use infer_train::inference::{Device, InferenceContext};
    use std::rc::Rc;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model: {}", model_path);

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    println!("✓ Model loaded");
    println!("  Layers: {}", ctx.layer_count());

    let tokens: Vec<u32> = vec![1, 2, 3];
    println!("\nInput tokens: {:?}", tokens);

    // Use the batched GPU forward (which uses GPU KV update kernels)
    println!("\n=== Testing batched GPU forward ===");
    match ctx.forward_gpu_batched(&tokens) {
        Some(logits) => {
            let num_nan = logits.iter().filter(|&&v| v.is_nan()).count();
            println!("Logits: {} values", logits.len());
            println!("NaN count: {}", num_nan);
            println!(
                "Range: [{:.4}, {:.4}]",
                logits.iter().cloned().fold(f32::INFINITY, f32::min),
                logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
            if num_nan == 0 {
                println!("✓ PASS: No NaN in output!");
            } else {
                println!("✗ FAIL: {} NaN values in output", num_nan);
            }
        }
        None => {
            println!("✗ FAIL: Forward pass returned None");
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This test requires Metal support");
}
