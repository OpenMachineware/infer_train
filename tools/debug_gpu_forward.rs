// Debug GPU forward pass - check intermediate results
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let model_path = "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    println!("Model: {}", model_path);

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");
    println!(
        "Config: hidden={}, heads={}, head_dim={}",
        ctx.config.hidden_size,
        ctx.config.n_heads,
        ctx.config.head_dim
    );

    // Test single token forward
    let tokens: Vec<u32> = vec![1];
    let logits = ctx.forward_gpu_batched(&tokens).expect("Forward failed");

    // Check output statistics
    let min = logits.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().sum();
    let count = logits.len();

    println!("\nLogits statistics:");
    println!("  Count: {}", count);
    println!("  Min: {:.4}", min);
    println!("  Max: {:.4}", max);
    println!("  Mean: {:.4}", sum / count as f32);
    println!("  First 10: {:?}", &logits[..10]);
    println!("  Last 10: {:?}", &logits[count-10..]);

    // Check if all zeros
    if logits.iter().all(|&x| x == 0.0) {
        println!("\nWARNING: All logits are zero!");
    }

    // Check for NaN
    if logits.iter().any(|&x| x.is_nan()) {
        println!("\nWARNING: Found NaN values!");
    }

    // Find top token
    let (top_idx, top_val) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, v)| (i, *v))
        .unwrap_or((0, 0.0));

    println!("\nTop token: {} (value: {:.4})", top_idx, top_val);
}
