// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Check model config differences

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let ctx = infer_train::inference::InferenceContext::from_gguf(
        model_path,
        infer_train::inference::Device::Cpu,
    )
    .expect("Failed to load model");

    println!("Architecture: {}", ctx.config.architecture);
    println!("Hidden size: {}", ctx.config.hidden_size);
    println!("Intermediate size: {}", ctx.config.intermediate_size);
    println!("n_layers: {}", ctx.config.n_layers);
    println!("n_heads: {}", ctx.config.n_heads);
    println!("n_kv_heads: {}", ctx.config.n_kv_heads);
    println!("head_dim: {}", ctx.config.head_dim);
    println!("vocab_size: {}", ctx.config.vocab_size);
    println!("rope_dim: {}", ctx.config.rope_dim);
    println!("rope_freq_base: {}", ctx.config.rope_freq_base);
    println!("rms_norm_eps: {}", ctx.config.rms_norm_eps);

    // Check layer shapes
    if let Some(layer) = ctx.model.layers.first() {
        println!("\nLayer 0 shapes:");
        if let Some(ref wq) = layer.wq {
            println!("  wq: {:?}", wq.shape);
        }
        if let Some(ref wk) = layer.wk {
            println!("  wk: {:?}", wk.shape);
        }
        if let Some(ref wv) = layer.wv {
            println!("  wv: {:?}", wv.shape);
        }
    }
}
