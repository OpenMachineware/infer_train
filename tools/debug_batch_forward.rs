// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug batch forward pass to trace where garbage output comes from

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
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
    let prompt = "Hello, how are you?";
    let tokens = tokenizer.encode(prompt);
    println!("\nPrompt: \"{}\"", prompt);
    println!("Tokens: {:?}", tokens);

    // Run debug forward pass
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (ctx.gpu_ctx.as_ref(), ctx.gpu_kv_cache.as_mut())
        {
            println!("\n=== Debug Forward Pass ===\n");
            let model = &ctx.model;

            // Call the debug version
            let logits =
                model.forward_prefill_batch_debug(&tokens, kv_cache, gpu_ctx);

            match logits {
                Some(l) => {
                    let max =
                        l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let min = l.iter().cloned().fold(f32::INFINITY, f32::min);
                    let num_nan = l.iter().filter(|&&v| v.is_nan()).count();
                    let top_token = l
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);

                    println!(
                        "\nFinal logits: min={:.2}, max={:.2}, nan={}",
                        min, max, num_nan
                    );
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
