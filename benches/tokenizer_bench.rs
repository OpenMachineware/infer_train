// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Tokenizer performance benchmark

use criterion::{
    BenchmarkId, Criterion, black_box, criterion_group, criterion_main,
};
use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
use std::hint::black_box as bb;

fn bench_tokenizer(c: &mut Criterion) {
    // Load tokenizers
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

    // Benchmark encode speed
    let mut group = c.benchmark_group("encode");
    for (name, text) in &texts {
        group.bench_with_input(
            BenchmarkId::new("devstral", name),
            text,
            |b, text| {
                b.iter(|| black_box(devstral.encode(text)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("qwen", name),
            text,
            |b, text| {
                b.iter(|| black_box(qwen.encode(text)));
            },
        );
    }
    group.finish();

    // Benchmark decode speed
    let mut group = c.benchmark_group("decode");
    for (name, text) in &texts {
        let devstral_tokens = devstral.encode(text);
        let qwen_tokens = qwen.encode(text);

        group.bench_with_input(
            BenchmarkId::new("devstral", name),
            &devstral_tokens,
            |b, tokens| {
                b.iter(|| black_box(devstral.decode(tokens)));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("qwen", name),
            &qwen_tokens,
            |b, tokens| {
                b.iter(|| black_box(qwen.decode(tokens)));
            },
        );
    }
    group.finish();

    // Print token compression ratios
    println!("\n=== Token Compression Ratio (chars/token) ===");
    println!(
        "{:<15} {:>12} {:>12} {:>12}",
        "Text", "Chars", "Devstral", "Qwen"
    );
    println!("{}", "-".repeat(55));
    for (name, text) in &texts {
        let devstral_tokens = devstral.encode(text);
        let qwen_tokens = qwen.encode(text);
        let chars = text.chars().count();
        let devstral_ratio = chars as f64 / (devstral_tokens.len() - 1) as f64; // -1 for BOS
        let qwen_ratio = chars as f64 / (qwen_tokens.len() - 1) as f64;

        println!(
            "{:<15} {:>12} {:>12.2} {:>12.2}",
            name, chars, devstral_ratio, qwen_ratio
        );
    }
}

criterion_group!(benches, bench_tokenizer);
criterion_main!(benches);
