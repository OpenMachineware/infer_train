// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test GPU forward pass with "Hello world" tokens

use infer_train::inference::{Device, InferenceContext};

const MODEL_PATH: &str =
    "/Users/jia/.llm_models/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf";

fn check_stats(name: &str, data: &[f32]) {
    let num_nan = data.iter().filter(|&&v| v.is_nan()).count();
    let num_inf = data.iter().filter(|&&v| v.is_infinite()).count();
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "  {}: len={}, nan={}, inf={}, range=[{:.4}, {:.4}]",
        name,
        data.len(),
        num_nan,
        num_inf,
        min,
        max
    );
}

fn main() {
    let tokens = vec![1u32, 22177, 4304]; // "Hello world" tokens
    println!("Tokens: {:?}", tokens);

    println!("\n=== GPU Forward ===");
    let mut ctx = InferenceContext::from_gguf(MODEL_PATH, Device::Gpu)
        .expect("Failed to load GPU context");

    let logits = ctx.forward_prefill_batch(&tokens).expect("Forward failed");
    check_stats("logits", &logits);
}
