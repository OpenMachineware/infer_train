// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug BPE merging

use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;

fn main() {
    let gguf = GgufFile::open("Qwen3.8-27B-UD-Q5_K_M.gguf").unwrap();
    let tokenizer = BpeTokenizer::from_gguf(&gguf).unwrap();

    // Test individual characters
    let chars = vec!["处", "理", "是", "人", "工", "智", "能"];

    println!("=== Individual character tokenization ===");
    for ch in chars {
        let tokens = tokenizer.encode(ch);
        println!("'{}' -> {:?}", ch, tokens);
    }

    // Test pairs
    println!("\n=== Pair tokenization ===");
    let pairs = vec![("处理", 96961), ("是", 95761), ("人工智能", 109015)];

    for (text, expected) in pairs {
        let tokens = tokenizer.encode(text);
        let found = tokens.iter().any(|&t| t == expected);
        println!(
            "'{}' (expected {}) -> {:?} {}",
            text,
            expected,
            tokens,
            if found { "✓" } else { "✗" }
        );
    }

    // Check if tokens exist in vocab
    println!("\n=== Vocab lookup ===");
    let test_tokens = vec!["是", "人工", "智能"];
    for t in test_tokens {
        let gpt2 = tokenizer.bytes_to_gpt2(t);
        let found = tokenizer.vocab_contains(&gpt2);
        println!(
            "'{}' -> '{}' in vocab: {}",
            t,
            gpt2,
            if found { "✓" } else { "✗" }
        );
    }
}
