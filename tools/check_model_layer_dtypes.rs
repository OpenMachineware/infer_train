// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Check weight dtypes for each layer in the model

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Model: {}", model_path);

    let ctx = infer_train::inference::InferenceContext::from_gguf(
        model_path,
        infer_train::inference::Device::Cpu,
    )
    .expect("Failed to load model");

    println!("\nModel architecture: {}", ctx.config.architecture);
    println!("Hidden size: {}", ctx.config.hidden_size);
    println!("Layers: {}", ctx.config.n_layers);
    println!("Vocab size: {}", ctx.config.vocab_size);

    println!("\n=== Layer weight dtypes ===");
    for (i, layer) in ctx.model.layers.iter().enumerate() {
        println!("\nLayer {}:", i);
        if let Some(ref wq) = layer.wq {
            println!("  wq: {:?} shape={:?}", wq.dtype, wq.shape);
        }
        if let Some(ref wk) = layer.wk {
            println!("  wk: {:?} shape={:?}", wk.dtype, wk.shape);
        }
        if let Some(ref wv) = layer.wv {
            println!("  wv: {:?} shape={:?}", wv.dtype, wv.shape);
        }
        if let Some(ref wo) = layer.wo {
            println!("  wo: {:?} shape={:?}", wo.dtype, wo.shape);
        }
        if let Some(ref w1) = layer.w1 {
            println!("  w1 (gate): {:?} shape={:?}", w1.dtype, w1.shape);
        }
        if let Some(ref w2) = layer.w2 {
            println!("  w2 (down): {:?} shape={:?}", w2.dtype, w2.shape);
        }
        if let Some(ref w3) = layer.w3 {
            println!("  w3 (up): {:?} shape={:?}", w3.dtype, w3.shape);
        }
    }

    if let Some(ref output) = ctx.model.output {
        println!("\n=== Output layer ===");
        println!("  dtype: {:?}", output.dtype);
        println!("  shape: {:?}", output.shape);
    }
}
