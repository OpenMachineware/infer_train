// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug KV cache dimensions

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    let model = &ctx.model;

    println!("Model config:");
    println!("  hidden_size: {}", model.config.hidden_size);
    println!("  head_dim: {}", model.config.head_dim);
    println!("  n_heads: {}", model.config.n_heads);
    println!("  n_kv_heads: {}", model.config.n_kv_heads);
    println!("  n_layers: {}", model.layers.len());

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let Some(ref mut kv_cache) = ctx.gpu_kv_cache {
            println!("\nKV cache:");
            println!("  max_seq: {}", kv_cache.max_seq());
            println!("  seq_len: {}", kv_cache.seq_len());

            let k_cache = kv_cache.k_buffer(0);
            println!("  K cache layer 0 size: {} bytes", k_cache.length());

            // Expected: n_kv_heads * head_dim * max_seq * sizeof(f32)
            let expected = model.config.n_kv_heads
                * model.config.head_dim
                * kv_cache.max_seq()
                * 4;
            println!("  Expected size: {} bytes", expected);

            // Check Layer 0 weight dimensions
            if let Some(wk) = &model.layers[0].wk {
                println!("\nLayer 0 K weight:");
                println!("  shape: {:?}", wk.shape);
                println!("  dtype: {:?}", wk.dtype);
            }
        }
    }
}
