// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Tokenizer test tool

use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return;
    }

    let gguf = GgufFile::open(&args[1]).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

    // Test encoding
    let test_text = "Hello, world!";
    let tokens = tokenizer.encode(test_text);
    println!("Input: \"{}\"", test_text);
    println!("Tokens: {:?}", tokens);

    // Test decoding
    let decoded = tokenizer.decode(&tokens);
    println!("Decoded: \"{}\"", decoded);

    // Test specific token IDs
    for &id in &[151646, 9707, 65797, 100, 1000, 10000] {
        let decoded = tokenizer.decode(&[id]);
        println!("Token {}: \"{}\"", id, decoded);
    }

    // Print vocab size
    println!("\nVocab size: {}", tokenizer.vocab_size());
}
