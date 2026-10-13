// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compare tokenizer output with llama.cpp

use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
use std::process::Command;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <text>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let text = &args[2];

    println!("Model: {}", model_path);
    println!("Text: \"{}\"", text);
    println!();

    // Our tokenizer
    println!("=== Our Tokenizer ===");
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

    let start = Instant::now();
    let our_tokens = tokenizer.encode(text);
    let our_time = start.elapsed();

    println!("Tokens ({}): {:?}", our_tokens.len(), our_tokens);
    println!("Time: {:?}", our_time);

    // llama.cpp tokenizer
    println!("\n=== llama.cpp Tokenizer ===");
    let start = Instant::now();
    let output = Command::new("llama-tokenize")
        .arg("-m")
        .arg(model_path)
        .arg("-p")
        .arg(text)
        .output()
        .expect("Failed to run llama-tokenize");
    let llama_time = start.elapsed();

    let llama_output = String::from_utf8_lossy(&output.stdout);
    println!("Output:\n{}", llama_output);
    println!("Time: {:?}", llama_time);

    // Parse llama.cpp output to extract tokens
    // Output format: "  1784 -> 'The'"
    let llama_tokens: Vec<u32> = llama_output
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            if l.contains("->") {
                l.split_whitespace().next().and_then(|n| n.parse().ok())
            } else {
                None
            }
        })
        .collect();

    // Compare
    println!("\n=== Comparison ===");
    println!("Our tokens:  {} {:?}", our_tokens.len(), our_tokens);
    println!("llama.cpp:   {} {:?}", llama_tokens.len(), llama_tokens);

    if our_tokens.len() == llama_tokens.len() {
        let matches: Vec<bool> = our_tokens
            .iter()
            .zip(llama_tokens.iter())
            .map(|(a, b)| a == b)
            .collect();
        let matching = matches.iter().filter(|&&m| m).count();
        println!(
            "\nMatching:    {}/{} ({:.1}%)",
            matching,
            our_tokens.len(),
            100.0 * matching as f64 / our_tokens.len() as f64
        );

        if matching != our_tokens.len() {
            println!("\nMismatches:");
            for (i, (a, b)) in
                our_tokens.iter().zip(llama_tokens.iter()).enumerate()
            {
                if a != b {
                    println!("  [{}] {} vs {}", i, a, b);
                }
            }
        } else {
            println!("\n✓ Perfect match!");
        }
    } else {
        println!("\n✗ Different token count!");
    }

    // Performance
    println!("\n=== Performance ===");
    println!("Our:       {:?}", our_time);
    println!("llama.cpp: {:?}", llama_time);
    let ratio = our_time.as_nanos() as f64 / llama_time.as_nanos() as f64;
    if ratio > 1.0 {
        println!("We are {:.2}x slower", ratio);
    } else {
        println!("We are {:.2}x faster", 1.0 / ratio);
    }
}
