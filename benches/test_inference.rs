// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// End-to-end inference test

use infer_train::inference::{dequant_to_f32, InferenceContext};
use infer_train::model::tensor::DataType;
use infer_train::ops::sampling::SamplingStrategy;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model: {}", model_path);

    // Load model
    let mut ctx = match InferenceContext::from_gguf(
        model_path,
        infer_train::inference::Device::Cpu,
    ) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            std::process::exit(1);
        }
    };

    println!("✓ Model loaded");

    let config = ctx.config.clone();
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Vocab size: {}", config.vocab_size);

    // Test embedding dequantization
    println!("\n=== Testing Embedding ===");
    if let Some(ref embed) = ctx.model.tok_embeddings {
        let dtype = embed.dtype;
        println!("  Embedding dtype: {:?}", dtype);
        println!("  Embedding shape: {:?}", embed.shape);

        let bytes = embed.bytes();
        println!("  Total bytes: {}", bytes.len());

        // Calculate bytes per row based on shape
        // Shape [1536, 151936] means 151936 rows of 1536 elements each
        let hidden_size = config.hidden_size;
        let rows = if embed.shape[0] == hidden_size {
            embed.shape[1]
        } else {
            embed.shape[0]
        };
        let cols = if embed.shape[0] == hidden_size {
            hidden_size
        } else {
            embed.shape[1]
        };

        println!("  Interpretation: {} rows x {} cols", rows, cols);

        let bytes_per_row = dtype.bytes_for_elems(cols);
        println!("  Bytes per row: {}", bytes_per_row);

        // Expected: rows * bytes_per_row
        let expected_bytes = rows * bytes_per_row;
        println!("  Expected total bytes: {}", expected_bytes);
        println!("  Actual total bytes: {}", bytes.len());

        // Print first few bytes of first row
        println!("\n  First 20 bytes of row 0:");
        for i in 0..20.min(bytes.len()) {
            print!("{:02x} ", bytes[i]);
        }
        println!();

        // Test embedding lookup for token 0
        println!("\n  Testing embedding lookup for token 0...");
        if let Some(hidden) = ctx.model.test_embedding_lookup(0) {
            println!("    Success! Got {} values", hidden.len());
            println!("    First 5: {:?}", &hidden[..5.min(hidden.len())]);
            let sum: f32 = hidden.iter().sum();
            println!("    Sum: {}", sum);
        } else {
            println!("    Failed: embedding_lookup returned None");
        }
    }

    // Check layers
    println!("\n=== Checking Layers ===");
    println!("  Layer count: {}", ctx.model.layers.len());
    if let Some(layer) = ctx.model.layers.first() {
        println!(
            "  Layer 0 has attention_norm: {}",
            layer.attention_norm.is_some()
        );
        println!("  Layer 0 has wq: {}", layer.wq.is_some());
        println!("  Layer 0 has wk: {}", layer.wk.is_some());
        println!("  Layer 0 has wv: {}", layer.wv.is_some());
        println!("  Layer 0 has wo: {}", layer.wo.is_some());
        println!("  Layer 0 has ffn_norm: {}", layer.ffn_norm.is_some());
        println!("  Layer 0 has w1: {}", layer.w1.is_some());
        println!("  Layer 0 has w2: {}", layer.w2.is_some());
        println!("  Layer 0 has w3: {}", layer.w3.is_some());
    }

    // Check output
    println!("\n=== Checking Output ===");
    println!("  Output norm: {}", ctx.model.norm.is_some());
    println!("  Output projection: {}", ctx.model.output.is_some());

    // Test generation
    println!("\n=== Testing Generation ===");
    let generated =
        ctx.generate(&[1, 2, 3, 4, 5], 5, &SamplingStrategy::Greedy);
    println!("✓ Generated {} tokens: {:?}", generated.len(), generated);

    println!("\n=== Done ===");
}
