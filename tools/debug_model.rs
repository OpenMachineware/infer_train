// Debug model loading
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::inference::Device;
use infer_train::inference::InferenceContext;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Model: {}", model_path);

    let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load GPU model");

    println!("\nConfig:");
    println!("  hidden_size: {}", ctx.config.hidden_size);
    println!("  n_heads: {}", ctx.config.n_heads);
    println!("  n_layers: {}", ctx.config.n_layers);
    println!("  head_dim: {}", ctx.config.head_dim);
    println!("  vocab_size: {}", ctx.config.vocab_size);

    println!("\nModel weights:");
    println!(
        "  tok_embeddings: {:?}",
        ctx.model.tok_embeddings.as_ref().map(|t| (t.dtype, &t.shape))
    );
    println!(
        "  norm: {:?}",
        ctx.model.norm.as_ref().map(|t| (t.dtype, &t.shape))
    );
    println!(
        "  output: {:?}",
        ctx.model.output.as_ref().map(|t| (t.dtype, &t.shape))
    );

    for (i, layer) in ctx.model.layers.iter().enumerate() {
        println!("\nLayer {}:", i);
        println!(
            "  attention_norm: {:?}",
            layer.attention_norm.as_ref().map(|t| (t.dtype, &t.shape))
        );
        println!("  wq: {:?}", layer.wq.as_ref().map(|t| (t.dtype, &t.shape)));
        println!("  wk: {:?}", layer.wk.as_ref().map(|t| (t.dtype, &t.shape)));
        println!("  wv: {:?}", layer.wv.as_ref().map(|t| (t.dtype, &t.shape)));
        println!("  wo: {:?}", layer.wo.as_ref().map(|t| (t.dtype, &t.shape)));
        println!(
            "  ffn_norm: {:?}",
            layer.ffn_norm.as_ref().map(|t| (t.dtype, &t.shape))
        );
        println!("  w1: {:?}", layer.w1.as_ref().map(|t| (t.dtype, &t.shape)));
        println!("  w2: {:?}", layer.w2.as_ref().map(|t| (t.dtype, &t.shape)));
        println!("  w3: {:?}", layer.w3.as_ref().map(|t| (t.dtype, &t.shape)));
    }
}
