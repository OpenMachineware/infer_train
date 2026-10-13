// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test tokenizer encoding/decoding

use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <text>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let text = &args[2];

    println!("Loading tokenizer from: {}", model_path);

    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

    println!("Vocab size: {}", tokenizer.vocab_size());
    println!(
        "BOS token: {} ({:?})",
        tokenizer.bos_token,
        tokenizer.token(tokenizer.bos_token)
    );
    println!(
        "EOS token: {} ({:?})",
        tokenizer.eos_token,
        tokenizer.token(tokenizer.eos_token)
    );
    println!("Add BOS: {}, Add EOS: {}", tokenizer.add_bos, tokenizer.add_eos);

    println!("\nEncoding: \"{}\"", text);
    let tokens = tokenizer.encode(text);
    println!("Tokens ({}): {:?}", tokens.len(), tokens);

    println!("\nDecoding back:");
    let decoded = tokenizer.decode(&tokens);
    println!("\"{}\"", decoded);

    // Show individual tokens
    println!("\nToken breakdown:");
    for &id in &tokens {
        if let Some(s) = tokenizer.token(id) {
            println!("  {} -> \"{}\"", id, s);
        }
    }
}
