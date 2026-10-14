// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Tokenizer performance test

use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
use std::time::Instant;

fn main() {
    // Load tokenizers
    println!("Loading tokenizers...");
    let devstral_gguf =
        GgufFile::open("Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf")
            .expect("Failed to load Devstral GGUF");
    let devstral = BpeTokenizer::from_gguf(&devstral_gguf)
        .expect("Failed to load Devstral tokenizer");

    let qwen_gguf = GgufFile::open("Qwen3.8-27B-UD-Q5_K_M.gguf")
        .expect("Failed to load Qwen GGUF");
    let qwen = BpeTokenizer::from_gguf(&qwen_gguf)
        .expect("Failed to load Qwen tokenizer");

    // Test texts
    let texts = vec![
        ("short_en", "The quick brown fox"),
        (
            "medium_en",
            "The quick brown fox jumps over the lazy dog. This is a longer sentence to test tokenization.",
        ),
        (
            "long_en",
            "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. It dates back to the 1950s and has evolved significantly with the advent of deep learning.",
        ),
        ("short_zh", "你好世界"),
        (
            "medium_zh",
            "自然语言处理是人工智能的重要分支，它涉及计算机与人类语言之间的交互。",
        ),
        (
            "long_zh",
            "深度学习是机器学习的一个分支，它试图使用包含复杂结构或由多重非线性变换构成的多个处理层对数据进行高层抽象的算法。深度学习在图像识别、语音识别、自然语言处理等领域取得了显著的成就。",
        ),
        ("mixed", "Hello 世界! This is a mixed 中英文 text 测试."),
        ("code", "fn main() {\n    println!(\"Hello, world!\");\n}"),
    ];

    println!("\n=== Token Compression Ratio (chars/token, excluding BOS) ===");
    println!("{:<15} {:>8} {:>12} {:>12}", "Text", "Chars", "Devstral", "Qwen");
    println!("{}", "-".repeat(50));

    let mut devstral_total_ratio = 0.0;
    let mut qwen_total_ratio = 0.0;
    let mut count = 0;

    for (name, text) in &texts {
        let devstral_tokens = devstral.encode(text);
        let qwen_tokens = qwen.encode(text);
        let chars = text.chars().count();
        // -1 to exclude BOS token
        let devstral_ratio =
            chars as f64 / (devstral_tokens.len() - 1).max(1) as f64;
        let qwen_ratio = chars as f64 / (qwen_tokens.len() - 1).max(1) as f64;

        println!(
            "{:<15} {:>8} {:>12.2} {:>12.2}",
            name, chars, devstral_ratio, qwen_ratio
        );

        devstral_total_ratio += devstral_ratio;
        qwen_total_ratio += qwen_ratio;
        count += 1;
    }

    println!("{}", "-".repeat(50));
    println!(
        "{:<15} {:>8} {:>12.2} {:>12.2}",
        "Average",
        "",
        devstral_total_ratio / count as f64,
        qwen_total_ratio / count as f64
    );

    // Performance test
    println!("\n=== Encode Performance (1000 iterations) ===");
    println!("{:<15} {:>12} {:>12}", "Text", "Devstral", "Qwen");
    println!("{}", "-".repeat(42));

    for (name, text) in &texts {
        // Devstral
        let start = Instant::now();
        for _ in 0..1000 {
            std::hint::black_box(devstral.encode(text));
        }
        let devstral_time = start.elapsed();

        // Qwen
        let start = Instant::now();
        for _ in 0..1000 {
            std::hint::black_box(qwen.encode(text));
        }
        let qwen_time = start.elapsed();

        println!(
            "{:<15} {:>10}ms {:>10}ms",
            name,
            devstral_time.as_millis(),
            qwen_time.as_millis()
        );
    }

    // Decode performance
    println!("\n=== Decode Performance (1000 iterations) ===");
    println!("{:<15} {:>12} {:>12}", "Text", "Devstral", "Qwen");
    println!("{}", "-".repeat(42));

    for (name, text) in &texts {
        let devstral_tokens = devstral.encode(text);
        let qwen_tokens = qwen.encode(text);

        // Devstral
        let start = Instant::now();
        for _ in 0..1000 {
            std::hint::black_box(devstral.decode(&devstral_tokens));
        }
        let devstral_time = start.elapsed();

        // Qwen
        let start = Instant::now();
        for _ in 0..1000 {
            std::hint::black_box(qwen.decode(&qwen_tokens));
        }
        let qwen_time = start.elapsed();

        println!(
            "{:<15} {:>10}ms {:>10}ms",
            name,
            devstral_time.as_millis(),
            qwen_time.as_millis()
        );
    }
}
