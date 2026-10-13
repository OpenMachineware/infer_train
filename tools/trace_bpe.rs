// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Detailed BPE trace

use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

fn main() {
    let gguf = GgufFile::open("Qwen3.8-27B-UD-Q5_K_M.gguf").unwrap();
    let tokenizer = BpeTokenizer::from_gguf(&gguf).unwrap();

    let text = "是";
    println!("Text: \"{}\"", text);

    // Manual BPE trace
    let bytes = tokenizer.bytes_to_gpt2(text);
    println!("GPT-2 encoding: \"{}\"", bytes);
    println!("Chars: {:?}", bytes.chars().collect::<Vec<_>>());

    let chars: Vec<char> = bytes.chars().collect();
    let n = chars.len();

    // Build initial tokens
    let mut tokens: Vec<(String, i32, i32)> = chars
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let prev = if i > 0 { (i - 1) as i32 } else { -1 };
            let next = if i < n - 1 { (i + 1) as i32 } else { -1 };
            (c.to_string(), prev, next)
        })
        .collect();

    println!("\nInitial tokens:");
    for (i, (t, prev, next)) in tokens.iter().enumerate() {
        println!("  [{}] '{}' (prev={}, next={})", i, t, prev, next);
    }

    // Find mergeable pairs
    println!("\nMergeable pairs:");
    for i in 0..n - 1 {
        if let Some(rank) =
            tokenizer.get_merge_rank(&tokens[i].0, &tokens[i + 1].0)
        {
            println!(
                "  [{}-{}] '{}' + '{}' -> rank {}",
                i,
                i + 1,
                tokens[i].0,
                tokens[i + 1].0,
                rank
            );
        }
    }

    // Show if final merged token exists in vocab
    let final_token = bytes.clone();
    println!(
        "\nFinal token '{}' in vocab: {}",
        final_token,
        tokenizer.vocab_contains(&final_token)
    );
}
