// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Layer 0 on CPU to verify model correctness

use infer_train::inference::{Device, InferenceContext, KVCache};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let model = &ctx.model;
    let hidden_size = model.config.hidden_size;

    // Get embedding
    let hidden = model.embedding_lookup_impl(9707).expect("No embedding");
    println!(
        "Embedding: {} values, first 5: {:?}",
        hidden.len(),
        &hidden[..5.min(hidden.len())]
    );

    // Create KV cache
    let mut kv_cache = KVCache::new(
        &model.config,
        2048,
        infer_train::inference::KVCacheFormat::F32,
    );

    // Get model config
    let n_heads = model.config.n_heads;
    let head_dim = model.config.head_dim;
    let n_kv_heads = model.config.n_kv_heads;

    // Run Layer 0 on CPU
    let layer0 = &model.layers[0];
    let result = model.forward_layer(
        layer0,
        0,
        &hidden,
        &mut kv_cache,
        n_heads,
        head_dim,
        n_kv_heads,
    );

    match result {
        Some(output) => {
            let nan_count = output.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "\nCPU Layer 0 output: {} NaN, first 5: {:?}",
                nan_count,
                &output[..5.min(output.len())]
            );

            if nan_count > 0 {
                let nan_positions: Vec<usize> = output
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_nan())
                    .map(|(i, _)| i)
                    .collect();
                println!(
                    "CPU NaN positions: {:?}",
                    &nan_positions[..10.min(nan_positions.len())]
                );
            }
        }
        None => {
            eprintln!("Layer 0 forward failed");
        }
    }
}
