// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Inspect tokenizer metadata in GGUF

use infer_train::model::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let gguf = GgufFile::open(&args[1]).expect("Failed to open GGUF");

    println!("=== Tokenizer Metadata ===");
    for (key, value) in &gguf.metadata {
        if key.starts_with("tokenizer.") {
            match value {
                infer_train::model::gguf::MetadataValue::String(s) => {
                    println!("{}: \"{}\"", key, s);
                }
                infer_train::model::gguf::MetadataValue::Uint64(v) => {
                    println!("{}: {}", key, v);
                }
                infer_train::model::gguf::MetadataValue::Int64(v) => {
                    println!("{}: {}", key, v);
                }
                infer_train::model::gguf::MetadataValue::Array(arr) => {
                    println!("{}: [array, {} items]", key, arr.len());
                    // Print first few items
                    for (i, item) in arr.iter().take(5).enumerate() {
                        match item {
                            infer_train::model::gguf::MetadataValue::String(
                                s,
                            ) => {
                                println!("  [{}]: \"{}\"", i, s);
                            }
                            _ => println!("  [{}]: {:?}", i, item),
                        }
                    }
                    if arr.len() > 5 {
                        println!("  ... and {} more", arr.len() - 5);
                    }
                }
                _ => println!("{}: {:?}", key, value),
            }
        }
    }
}
