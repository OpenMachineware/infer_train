// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug tokenizer loading

use infer_train::model::{GgufFile, MetadataValue};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return;
    }

    let gguf = GgufFile::open(&args[1]).expect("Failed to open GGUF");
    let meta = &gguf.metadata;

    // Check tokens
    println!("Checking tokenizer.ggml.tokens...");
    match meta.get("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array(arr)) => {
            println!("  Array length: {}", arr.len());
            if let Some(MetadataValue::String(s)) = arr.get(0) {
                println!("  First token: {}", s);
            } else if let Some(v) = arr.get(0) {
                println!("  First element is not a string: {:?}", std::mem::discriminant(v));
            } else {
                println!("  Array is empty");
            }
        }
        Some(other) => println!("  Not an array: {:?}", std::mem::discriminant(other)),
        None => println!("  Key not found"),
    }

    // Check merges
    println!("\nChecking tokenizer.ggml.merges...");
    match meta.get("tokenizer.ggml.merges") {
        Some(MetadataValue::Array(arr)) => {
            println!("  Array length: {}", arr.len());
            if let Some(MetadataValue::String(s)) = arr.get(0) {
                println!("  First merge: {}", s);
            } else if let Some(v) = arr.get(0) {
                println!("  First element is not a string: {:?}", std::mem::discriminant(v));
            } else {
                println!("  Array is empty");
            }
        }
        Some(other) => println!("  Not an array: {:?}", std::mem::discriminant(other)),
        None => println!("  Key not found"),
    }

    // Check model type
    println!("\nChecking tokenizer.ggml.model...");
    match meta.get("tokenizer.ggml.model") {
        Some(MetadataValue::String(s)) => println!("  Model: {}", s),
        Some(_) => println!("  Not a string"),
        None => println!("  Key not found"),
    }

    // Check pre type
    println!("\nChecking tokenizer.ggml.pre...");
    match meta.get("tokenizer.ggml.pre") {
        Some(MetadataValue::String(s)) => println!("  Pre: {}", s),
        Some(_) => println!("  Not a string"),
        None => println!("  Key not found"),
    }
}
