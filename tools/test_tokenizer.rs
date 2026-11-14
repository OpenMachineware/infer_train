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

    // Create tokenizer
    let tokenizer = create_tokenizer(meta).expect("Failed to create tokenizer");
    println!("\nTokenizer created successfully!");

    // Test various text types
    let test_texts = [
        "Hello, world!",
        "你好世界",
        "Hello 你好 World 世界",
        "测试123test",
        "这是一个很长的中文句子，用来测试tokenizer对长文本的处理能力。",
        "The quick brown fox jumps over the lazy dog.",
        "日本語テスト",
        "한국어 테스트",
        "🎉🎊🎁", // emoji
    ];

    println!("\n=== Comprehensive tests ===");
    let mut all_passed = true;
    for text in &test_texts {
        println!("\nText: '{}'", text);
        let tokens = tokenizer.encode(text).expect("Failed to encode");
        println!("Tokens: {:?}", tokens);
        let decoded =
            tokenizer.decode_skip_special(&tokens).expect("Failed to decode");
        println!("Decoded: '{}'", decoded);
        if text == &decoded {
            println!("✓ Perfect match!");
        } else {
            println!("✗ Mismatch!");
            all_passed = false;
        }
    }

    if all_passed {
        println!("\n=== All tests passed! ===");
    } else {
        println!("\n=== Some tests failed! ===");
        std::process::exit(1);
    }
}
