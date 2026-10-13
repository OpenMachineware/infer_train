// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Trace BPE for a sequence

use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = if args.len() > 1 { &args[1] } else { "处理是" };

    let gguf = GgufFile::open("Qwen3.8-27B-UD-Q5_K_M.gguf").unwrap();
    let tokenizer = BpeTokenizer::from_gguf(&gguf).unwrap();

    println!("Text: \"{}\"", text);

    // Show individual tokenizations
    println!("\n=== Individual ===");
    for ch in text.chars() {
        let tokens = tokenizer.encode(&ch.to_string());
        println!("'{}' -> {:?}", ch, tokens);
    }

    // Show combined
    println!("\n=== Combined ===");
    let tokens = tokenizer.encode(text);
    println!("'{}' -> {:?}", text, tokens);

    // Show token breakdown
    println!("\n=== Token breakdown ===");
    for &id in &tokens {
        if let Some(s) = tokenizer.token(id) {
            println!("  {} -> \"{}\"", id, s);
        }
    }
}
