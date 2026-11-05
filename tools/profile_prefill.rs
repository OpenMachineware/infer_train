// Profile prefill performance
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <prompt>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let prompt = &args[2];

    println!("Loading model: {}", model_path);
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    println!("GPU context: {}", ctx.gpu_ctx.is_some());
    println!("GPU KV cache: {}", ctx.gpu_kv_cache.is_some());
    println!("Tok embeddings: {}", ctx.model.tok_embeddings.is_some());

    // Tokenize
    let tokenizer = infer_train::tokenize::BpeTokenizer::from_gguf(
        &infer_train::model::GgufFile::open(model_path).unwrap(),
    )
    .expect("Failed to load tokenizer");
    let tokens = tokenizer.encode(prompt);
    println!("\nPrompt: \"{}\"", prompt);
    println!(
        "Tokens: {} ({:?})",
        tokens.len(),
        &tokens[..std::cmp::min(5, tokens.len())]
    );

    // Profile prefill
    println!("\n=== Prefill ===");
    let start = std::time::Instant::now();

    let result = ctx.forward_prefill_batch(&tokens);

    let elapsed = start.elapsed();
    println!("Time: {:?}", elapsed);
    println!(
        "Result: {:?}",
        result.as_ref().map(|v| format!("{} logits", v.len()))
    );

    if let Some(logits) = result {
        let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
        println!("NaN count: {}", nan_count);
        println!("First 5: {:?}", &logits[..5]);
    }
}
