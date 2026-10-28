// SPDX-License-Identifier: AGPL-3.0-or-later
use infer_train::inference::{Device, InferenceContext, KVCache, KVCacheFormat};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let mut ctx = InferenceContext::from_gguf(&args[1], Device::Cpu)
        .expect("Failed to load model");

    // Check output tensor
    if let Some(ref output) = ctx.model.output {
        println!("Output tensor:");
        println!("  dtype: {:?}", output.dtype);
        println!("  shape: {:?}", output.shape);
        println!("  bytes len: {}", output.bytes().len());
    } else {
        println!("Output tensor is None!");
    }

    // Check norm tensor
    if let Some(ref norm) = ctx.model.norm {
        println!("Final norm tensor:");
        println!("  dtype: {:?}", norm.dtype);
        println!("  shape: {:?}", norm.shape);
    } else {
        println!("Final norm tensor is None!");
    }

    // Try a forward pass and trace each step
    let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);

    println!("\n=== Forward step by step ===");

    // 1. Embedding
    let mut hidden = ctx.model.embedding_lookup_impl(1).expect("Embedding failed");
    println!("After embedding: sum = {:.4}", hidden.iter().sum::<f32>());

    // 2. Process through layers
    for (layer_idx, layer) in ctx.model.layers.iter().enumerate() {
        let result = ctx.model.forward_layer(
            layer, layer_idx, &hidden, &mut kv_cache,
            ctx.config.n_heads, ctx.config.head_dim, ctx.config.n_kv_heads
        );

        match result {
            Some(new_hidden) => {
                let sum: f32 = new_hidden.iter().sum();
                if sum == 0.0 {
                    println!("Layer {} returned all zeros!", layer_idx);
                }
                hidden = new_hidden;
                if layer_idx < 3 {
                    println!("After layer {}: sum = {:.4}", layer_idx, sum);
                }
            }
            None => {
                println!("Layer {} forward_layer returned None!", layer_idx);
                return;
            }
        }
    }

    println!("After all layers: sum = {:.4}", hidden.iter().sum::<f32>());

    // 3. Final norm
    let final_normed = ctx.model.apply_output_norm(&hidden);
    match final_normed {
        Some(n) => {
            println!("After final norm: sum = {:.4}", n.iter().sum::<f32>());
        }
        None => {
            println!("Final norm returned None!");
            return;
        }
    }

    // 4. Output projection
    let logits = ctx.model.output_projection(&final_normed.unwrap(), ctx.config.vocab_size);
    match logits {
        Some(l) => {
            println!("Output projection: len = {}, sum = {:.4}, non-zero = {}",
                l.len(), l.iter().sum::<f32>(), l.iter().filter(|&&x| x != 0.0).count());
        }
        None => {
            println!("Output projection returned None!");
        }
    }
}
