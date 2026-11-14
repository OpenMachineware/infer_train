// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Tokenizer test with real GGUF file

use infer_train::model::GgufFile;
use infer_train::tokenizer::create_tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = if args.len() > 1 {
        args[1].as_str()
    } else {
        "Qwen3-0.6B-UD-Q4_K_XL.gguf"
    };

    println!("Loading {}...", path);

    let gguf = GgufFile::load(path).expect("Failed to load GGUF");
    println!("GGUF version: {}", gguf.version);
    println!("Number of tensors: {}", gguf.tensors.len());
    println!("Number of KV pairs: {}", gguf.kv.len());

    // Extract tokenizer metadata
    let meta =
        gguf.tokenizer_metadata().expect("Failed to get tokenizer metadata");
    println!("\nTokenizer type: {:?}", meta.tokenizer_type);
    println!("Vocab size: {}", meta.tokens.len());
    println!("Number of merges: {}", meta.merges.len());
    println!("BOS token ID: {:?}", meta.bos_token_id);
    println!("EOS token ID: {:?}", meta.eos_token_id);
    println!("UNK token ID: {:?}", meta.unk_token_id);
    println!("Add BOS: {:?}", meta.add_bos);
    println!("Add EOS: {:?}", meta.add_eos);
    println!("Byte fallback: {:?}", meta.byte_fallback);
    println!("Pre: {:?}", meta.pre);

    // Show first 10 tokens
    println!("\nFirst 10 tokens:");
    for (i, token) in meta.tokens.iter().take(10).enumerate() {
        println!("  {} -> '{}'", i, token);
    }

    // Show first 5 merges
    if !meta.merges.is_empty() {
        println!("\nFirst 5 merges:");
        for (t1, t2) in meta.merges.iter().take(5) {
            println!("  '{}' + '{}' -> '{}{}'", t1, t2, t1, t2);
        }
    }

    // Create tokenizer
    let tokenizer = create_tokenizer(meta).expect("Failed to create tokenizer");
    println!("\nTokenizer created successfully!");

    // Test encoding
    let text = "Hello, world!";
    println!("\nEncoding: '{}'", text);
    let tokens = tokenizer.encode(text).expect("Failed to encode");
    println!("Tokens: {:?}", tokens);

    // Decode back
    let decoded = tokenizer.decode(&tokens).expect("Failed to decode");
    println!("Decoded: '{}'", decoded);

    // Decode skip special
    let decoded_skip = tokenizer
        .decode_skip_special(&tokens)
        .expect("Failed to decode skip special");
    println!("Decoded (skip special): '{}'", decoded_skip);
}
