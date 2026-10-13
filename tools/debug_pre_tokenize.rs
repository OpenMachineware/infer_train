// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug pre-tokenization

use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
use regex::Regex;

fn main() {
    let gguf = GgufFile::open("Qwen3.8-27B-UD-Q5_K_M.gguf").unwrap();
    let tokenizer = BpeTokenizer::from_gguf(&gguf).unwrap();

    println!("Pre-type: {:?}", tokenizer.pre_type);

    let text = "处理是人工智能";
    println!("\nText: \"{}\"", text);

    // Test pre-tokenization regex
    let re = Regex::new(
        r"(?:'[sdmt]|'ll|'ve|'re)|(?:[^\r\n\p{L}\p{N}]?\p{L}+)|(?:\p{N}{1,3})|(?:[^\s\p{L}\p{N}]+)|(?:\s+)",
    ).unwrap();

    println!("\nPre-tokenization chunks:");
    for m in re.find_iter(text) {
        println!("  \"{}\"", m.as_str());
    }

    // Show final tokens
    let tokens = tokenizer.encode(text);
    println!("\nFinal tokens: {:?}", tokens);
}
