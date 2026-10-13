// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test tokenizer with long text to verify O(n log n) performance

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

    // Test with increasingly long text
    let test_lengths =
        [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000];

    // Load tokenizer
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

    println!("Testing tokenizer with various text lengths:");
    println!("{:<10} {:<12} {:<12}", "Chars", "Tokens", "Time (ms)");
    println!("{}", "-".repeat(35));

    for n in test_lengths {
        let text =
            "The quick brown fox jumps over the lazy dog. ".repeat(n / 50 + 1);
        let text: String = text.chars().take(n).collect();

        let start = Instant::now();
        let tokens = tokenizer.encode(&text);
        let elapsed = start.elapsed();

        println!(
            "{:<10} {:<12} {:<12.3}",
            n,
            tokens.len(),
            elapsed.as_secs_f64() * 1000.0
        );
    }

    // Test with Chinese text
    println!("\nTesting with Chinese text:");
    println!("{:<10} {:<12} {:<12}", "Chars", "Tokens", "Time (ms)");
    println!("{}", "-".repeat(35));

    for n in test_lengths {
        let text = "人工智能正在改变世界。".repeat(n / 10 + 1);
        let text: String = text.chars().take(n).collect();

        let start = Instant::now();
        let tokens = tokenizer.encode(&text);
        let elapsed = start.elapsed();

        println!(
            "{:<10} {:<12} {:<12.3}",
            n,
            tokens.len(),
            elapsed.as_secs_f64() * 1000.0
        );
    }
}
