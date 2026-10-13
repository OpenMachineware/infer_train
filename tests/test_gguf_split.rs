// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test GGUF split file loading
// Run: cargo run --release --bin test_gguf_split -- DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf-00001-of-00003.gguf

use infer_train::model::{GgufFile, WeightStore};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <split-gguf-file-part-1>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    println!("Testing split file: {}", path);

    // Test GGUF split detection
    let gguf = match GgufFile::open(path) {
        Ok(gguf) => gguf,
        Err(e) => {
            eprintln!("✗ Error opening file: {}", e);
            std::process::exit(1);
        }
    };

    println!("✓ GGUF version: {}", gguf.version);
    println!("✓ Is split: {}", gguf.is_split());

    // Print split-related metadata
    for key in &["split.count", "split.no"] {
        if let Some(val) = gguf.metadata.get(*key) {
            println!("  Metadata {}: {:?}", key, val);
        }
    }

    if gguf.is_split() {
        if let Some((no, count)) = gguf.split_info() {
            println!("✓ Split info: part {} of {}", no, count);
        } else {
            println!("✗ split_info() returned None");
        }
    }

    println!();

    // Test split loading via WeightStore
    println!("=== Loading Split Weights ===");
    let store = match WeightStore::from_gguf(path) {
        Ok(store) => store,
        Err(e) => {
            eprintln!("✗ WeightStore error: {}", e);
            std::process::exit(1);
        }
    };

    println!("✓ Loaded {} tensors", store.tensor_count());
    println!(
        "✓ Total size: {:.2} MB",
        store.total_bytes() as f64 / 1024.0 / 1024.0
    );

    // Print first few tensor names
    println!("\n=== Sample Tensors ===");
    for (i, name) in store.tensor_names().iter().enumerate().take(20) {
        println!("  [{}] {}", i, name);
    }
}
