// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug output layer and F16 GEMV

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = &args[1];

    println!("Loading model: {}", model_path);

    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let model = ctx.model();

    // Check output weight
    println!("\n=== Output Weight Analysis ===");
    if let Some(output) = &model.output {
        println!("Output tensor:");
        println!("  dtype: {:?}", output.dtype);
        println!("  shape: {:?}", output.shape);
        println!("  bytes len: {}", output.bytes().len());

        let is_transposed = output.shape.len() == 2
            && output.shape[0] == ctx.config.hidden_size;
        println!("  is_transposed: {}", is_transposed);

        let actual_vocab_size =
            if is_transposed { output.shape[1] } else { output.shape[0] };
        println!("  actual_vocab_size: {}", actual_vocab_size);
        println!("  config.vocab_size: {}", ctx.config.vocab_size);

        // Check the first few bytes of the weight
        println!("\n  First 40 bytes:");
        for (i, chunk) in output.bytes().chunks(4).take(10).enumerate() {
            println!("    [{}]: {:02x?}", i, chunk);
        }
    }

    // Check the final norm
    println!("\n=== Final Norm Analysis ===");
    if let Some(norm) = &model.norm {
        println!("Norm tensor:");
        println!("  dtype: {:?}", norm.dtype);
        println!("  shape: {:?}", norm.shape);
        println!("  bytes len: {}", norm.bytes().len());
    }

    // Check config
    println!("\n=== Model Config ===");
    println!("  hidden_size: {}", ctx.config.hidden_size);
    println!("  vocab_size: {}", ctx.config.vocab_size);
    println!("  n_layers: {}", ctx.config.n_layers);
    println!("  n_heads: {}", ctx.config.n_heads);
    println!("  n_kv_heads: {}", ctx.config.n_kv_heads);
    println!("  head_dim: {}", ctx.config.head_dim);
}
