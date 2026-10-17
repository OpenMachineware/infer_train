// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test the new GPU batched forward path

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
use std::rc::Rc;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    // Load tokenizer
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

    // Load model
    println!("Loading model: {}", model_path);
    let start = Instant::now();
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");
    println!("Model loaded in {:?}", start.elapsed());
    println!("GPU context available: {}", ctx.gpu_ctx.is_some());
    println!("GPU KV cache available: {}", ctx.gpu_kv_cache.is_some());

    // Test prompt
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt);
    println!("\nPrompt: \"{}\"", prompt);
    println!("Tokens: {:?}", tokens);

    // Test new GPU path
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (ctx.gpu_ctx.as_ref(), ctx.gpu_kv_cache.as_mut())
        {
            println!("\n=== Testing new GPU batched path ===\n");
            let model = &ctx.model;

            // Test with just 2 tokens first
            let test_tokens = &tokens[0..2.min(tokens.len())];
            println!(
                "Testing with {} tokens: {:?}",
                test_tokens.len(),
                test_tokens
            );

            let start = Instant::now();
            match model.forward_gpu_native_batched(
                test_tokens,
                kv_cache,
                gpu_ctx,
            ) {
                Some(logits) => {
                    let elapsed = start.elapsed();
                    println!("\nForward pass completed in {:?}", elapsed);

                    let max = logits
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let min =
                        logits.iter().cloned().fold(f32::INFINITY, f32::min);
                    let num_nan =
                        logits.iter().filter(|&&v| v.is_nan()).count();
                    let num_zero = logits.iter().filter(|&&v| v == 0.0).count();

                    println!(
                        "Logits: len={}, min={:.2}, max={:.2}, nan={}, zero={}",
                        logits.len(),
                        min,
                        max,
                        num_nan,
                        num_zero
                    );

                    if num_zero > logits.len() / 2 {
                        println!("WARNING: More than half of logits are zero!");
                    }

                    let top_token = logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);

                    println!(
                        "Top token: {} ({:?})",
                        top_token,
                        tokenizer.decode(&[top_token as u32])
                    );
                }
                None => {
                    println!("Forward pass failed!");
                }
            }
        } else {
            println!("GPU context not available");
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("GPU support not available");
    }
}
