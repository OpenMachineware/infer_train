// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

//! Quick CLI testing interface

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};
use infer_train::model::GgufFile;
use infer_train::ops::sample_token;
use infer_train::tokenize::BpeTokenizer;
use std::io::{self, BufRead, Write};

pub fn run(
    model_path: &str,
    device: &str,
    max_tokens: usize,
    temperature: f32,
    prompt: Option<&str>,
) {
    let device = match device.to_lowercase().as_str() {
        "cpu" => Device::Cpu,
        "gpu" => Device::Gpu,
        _ => {
            eprintln!("Invalid device '{}', using GPU", device);
            Device::Gpu
        }
    };

    println!("Loading model from: {}", model_path);

    // Load tokenizer
    let gguf = match GgufFile::open(model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to open GGUF: {}", e);
            return;
        }
    };
    let tokenizer = match BpeTokenizer::from_gguf(&gguf) {
        Some(t) => t,
        None => {
            eprintln!("Failed to load tokenizer");
            return;
        }
    };

    let ctx = match InferenceContext::from_gguf(model_path, device) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            return;
        }
    };

    println!(
        "Model loaded: {} layers, {} hidden, {} heads",
        ctx.model.layers.len(),
        ctx.config.hidden_size,
        ctx.config.n_heads
    );
    println!("Device: {:?}", device);
    println!("Temperature: {:.2}", temperature);
    println!("Max tokens: {}", max_tokens);
    println!();

    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    let input = match prompt {
        Some(p) => p.to_string(),
        None => {
            print!("Enter prompt: ");
            io::stdout().flush().unwrap();
            let mut line = String::new();
            io::stdin().lock().read_line(&mut line).unwrap();
            line.trim().to_string()
        }
    };

    if input.is_empty() {
        println!("No input provided, exiting.");
        return;
    }

    // Tokenize input
    let tokens = tokenizer.encode(&input);

    println!("Input tokens: {}", tokens.len());

    // Prefill
    print!("Generating: {}", input);
    io::stdout().flush().unwrap();

    let mut generated = 0;
    let mut current_tokens = tokens.clone();

    while generated < max_tokens {
        let logits = match ctx.model.forward(&current_tokens, &mut kv_cache) {
            Some(l) => l,
            None => {
                eprintln!("\nForward pass failed");
                break;
            }
        };

        // Sample next token
        let next_token = sample_token(&logits, temperature) as u32;
        current_tokens.clear();
        current_tokens.push(next_token);

        // Decode token to text
        let text = tokenizer.decode(&[next_token]);
        print!("{}", text);
        io::stdout().flush().unwrap();

        generated += 1;
    }

    println!("\n\nGenerated {} tokens", generated);
}
