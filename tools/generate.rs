// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Text generation tool

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use infer_train::ops::sampling::sample_greedy;
use infer_train::tokenize::BpeTokenizer;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <prompt> [max_tokens]", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let prompt = &args[2];
    let max_tokens: usize =
        args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);

    println!("Loading model: {}", model_path);
    let start_load = Instant::now();

    // Load tokenizer
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");
    println!("Tokenizer loaded in {:?}", start_load.elapsed());

    // Load model on GPU
    let start_model = Instant::now();
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");
    println!("Model loaded in {:?}", start_model.elapsed());
    println!("  Layers: {}", ctx.layer_count());
    println!("  Params: {}M", ctx.param_count() / 1_000_000);

    // Tokenize prompt
    let prompt_tokens = tokenizer.encode(prompt);
    println!("\nPrompt: \"{}\"", prompt);
    println!("Tokens: {} ({:?})", prompt_tokens.len(), prompt_tokens);

    // Prefill using batch GEMM (GPU) or CPU forward
    println!("\n=== Prefill (batch) ===");
    let start_prefill = Instant::now();

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    let logits =
        ctx.forward_prefill_batch(&prompt_tokens).expect("Prefill failed");

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    let logits = {
        let mut kv_cache = infer_train::inference::KVCache::new(
            &ctx.config,
            prompt_tokens.len() + max_tokens,
            infer_train::inference::KVCacheFormat::F32,
        );
        ctx.forward(&prompt_tokens, &mut kv_cache).expect("Prefill failed")
    };

    let prefill_time = start_prefill.elapsed();
    println!("Prefill time: {:?}", prefill_time);
    println!(
        "Tokens/sec: {:.2}",
        prompt_tokens.len() as f64 / prefill_time.as_secs_f64()
    );

    // Generate tokens
    println!("\n=== Generation (max {} tokens) ===", max_tokens);
    let mut generated_tokens = Vec::new();
    let mut current_tokens = prompt_tokens.clone();
    let start_gen = Instant::now();

    // Create KV cache for CPU fallback
    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    let mut kv_cache = infer_train::inference::KVCache::new(
        &ctx.config,
        prompt_tokens.len() + max_tokens,
        infer_train::inference::KVCacheFormat::F32,
    );

    for i in 0..max_tokens {
        // Get logits for last token
        let decode_logits;
        let logits = if i == 0 {
            &logits
        } else {
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            {
                decode_logits = ctx
                    .forward_gpu_native(&[
                        current_tokens[current_tokens.len() - 1]
                    ])
                    .expect("Decode failed");
                &decode_logits
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
            {
                decode_logits = ctx
                    .forward(
                        &[current_tokens[current_tokens.len() - 1]],
                        &mut kv_cache,
                    )
                    .expect("Decode failed");
                &decode_logits
            }
        };

        // Sample next token
        let next_token = sample_greedy(logits) as u32;

        // Debug: print logits stats
        if i == 0 {
            let max_logit =
                logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_logit =
                logits.iter().cloned().fold(f32::INFINITY, f32::min);
            let num_nan = logits.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "  First token logits: min={:.2}, max={:.2}, nan={}, sample={}",
                min_logit, max_logit, num_nan, next_token
            );
        }

        // Check EOS
        if next_token == tokenizer.eos_token {
            println!("  <EOS>");
            break;
        }

        // Decode and print
        let token_str = tokenizer.decode(&[next_token]);
        print!("{}", token_str);
        use std::io::Write;
        std::io::stdout().flush().ok();

        generated_tokens.push(next_token);
        current_tokens.push(next_token);
    }

    let gen_time = start_gen.elapsed();
    println!("\n\n=== Stats ===");
    println!("Generated: {} tokens", generated_tokens.len());
    println!("Time: {:?}", gen_time);
    println!(
        "Tokens/sec: {:.2}",
        generated_tokens.len() as f64 / gen_time.as_secs_f64()
    );
}
