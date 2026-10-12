// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test unified model loader with architecture detection
// Run: cargo run --release --bin test_unified -- model.gguf

use infer_train::model::{Architecture, Model};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf-file>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    println!("Loading model: {}", path);

    let model = Model::from_gguf(path).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    });

    println!("=== Architecture ===");
    println!("  Type: {:?}", model.architecture);
    println!("  Name: {}", model.config.architecture);
    println!();

    println!("=== Config ===");
    println!("  Hidden: {}", model.config.hidden_size);
    println!("  Layers: {}", model.config.n_layers);
    println!(
        "  Heads: {} (KV: {})",
        model.config.n_heads, model.config.n_kv_heads
    );
    println!();

    println!("=== Layers ===");
    println!("  Llama: {}", model.llama_layers.len());
    println!("  MoE: {}", model.moe_layers.len());
    println!("  Mamba: {}", model.mamba_layers.len());
    println!("  Total: {}", model.layer_count());
    println!("  Complete: {}/{}", model.complete_layers(), model.layer_count());
    println!();

    println!("=== Parameters ===");
    println!("  Total: {:.2}B", model.param_count() as f64 / 1e9);
}
