// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test GGUF parsing and model layer
// Run: cargo run --release --bin test_gguf -- models/qwen-7b-q4_0.gguf

use infer_train::model::{GgufFile, LlamaModel, ModelConfig, WeightStore};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <gguf-file>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    println!("Loading GGUF file: {}", path);

    // Test GGUF parsing
    let gguf = match GgufFile::open(path) {
        Ok(gguf) => gguf,
        Err(e) => {
            eprintln!("✗ Error: {}", e);
            std::process::exit(1);
        }
    };

    println!("✓ GGUF version: {}", gguf.version);
    println!("✓ Tensor count: {}", gguf.tensor_count);
    println!();

    // Test ModelConfig
    let config = ModelConfig::from_gguf(&gguf);
    println!("=== Model Config ===");
    println!("  Architecture: {}", config.architecture);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Intermediate size: {}", config.intermediate_size);
    println!("  Layers: {}", config.n_layers);
    println!("  Heads: {} (KV: {})", config.n_heads, config.n_kv_heads);
    println!("  Vocab size: {}", config.vocab_size);
    println!(
        "  RoPE dim: {} (base: {})",
        config.rope_dim, config.rope_freq_base
    );
    println!("  RMS norm eps: {}", config.rms_norm_eps);
    println!("  Max position: {}", config.max_position_embeddings);
    println!("  Head dim: {}", config.head_dim());
    println!();

    // Test WeightStore
    println!("=== Loading Weights ===");
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
    println!();

    // Print ALL tensor names to understand the naming convention
    println!("=== Tensor Names (first 50) ===");
    for (i, name) in store.tensor_names().iter().enumerate().take(50) {
        println!("  [{}] {}", i, name);
    }
    println!();

    // Test Llama Model
    println!("=== Building Llama Model ===");
    let model = LlamaModel::from_weights(&store, config.clone());
    println!("✓ Loaded {} layers", model.layer_count());

    // Check layer completeness
    let complete_layers =
        model.layers.iter().filter(|l| l.is_complete()).count();
    println!("✓ Complete layers: {}/{}", complete_layers, model.layer_count());

    // Print sample layer info
    if let Some(layer) = model.layers.first() {
        println!("\n=== Layer 0 Weights ===");
        if let Some(ref t) = layer.attention_norm {
            println!("  attention_norm: {:?} {:?}", t.shape, t.dtype);
        }
        if let Some(ref t) = layer.wq {
            println!("  wq: {:?} {:?}", t.shape, t.dtype);
        }
        if let Some(ref t) = layer.wk {
            println!("  wk: {:?} {:?}", t.shape, t.dtype);
        }
        if let Some(ref t) = layer.wv {
            println!("  wv: {:?} {:?}", t.shape, t.dtype);
        }
        if let Some(ref t) = layer.wo {
            println!("  wo: {:?} {:?}", t.shape, t.dtype);
        }
        if let Some(ref t) = layer.w1 {
            println!("  w1 (gate): {:?} {:?}", t.shape, t.dtype);
        }
        if let Some(ref t) = layer.w2 {
            println!("  w2 (down): {:?} {:?}", t.shape, t.dtype);
        }
        if let Some(ref t) = layer.w3 {
            println!("  w3 (up): {:?} {:?}", t.shape, t.dtype);
        }
    }

    // Print parameter count
    let param_count = model.param_count();
    println!("\n=== Total Parameters: {:.2}B ===", param_count as f64 / 1e9);
}
