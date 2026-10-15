// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug layer 3 NaN - full layer processing

fn main() {
    println!("Layer 3 full debug\n");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{
            InferenceContext, KVCache, KVCacheFormat,
        };

        let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";

        // Load model
        println!("Loading model on CPU...");
        let ctx = InferenceContext::from_gguf(
            model_path,
            infer_train::inference::Device::Cpu,
        )
        .expect("Failed to load model");

        let model = ctx.model();
        let config = ctx.config.clone();

        // Create CPU KV cache
        let mut kv_cache = KVCache::new(&config, 16, KVCacheFormat::F32);

        // Process token 1
        let hidden = model.embedding_lookup_impl(1).expect("Embedding failed");
        let mut hidden = hidden;

        // Process layers 0-2
        for (layer_idx, layer) in model.layers.iter().take(3).enumerate() {
            hidden = model
                .forward_layer(
                    layer,
                    layer_idx,
                    &hidden,
                    &mut kv_cache,
                    config.n_heads,
                    config.head_dim,
                    config.n_kv_heads,
                )
                .expect(&format!("Layer {} failed", layer_idx));
        }

        println!("✓ Processed layers 0-2");

        // Process layer 3
        println!("\n=== Processing layer 3 ===\n");
        let layer = &model.layers[3];
        let result = model.forward_layer(
            layer,
            3,
            &hidden,
            &mut kv_cache,
            config.n_heads,
            config.head_dim,
            config.n_kv_heads,
        );

        match result {
            Some(ref out) => {
                let num_nan: usize =
                    out.iter().filter(|&&v| v.is_nan()).count();
                println!(
                    "Layer 3 output: {} values, {} NaN, first 3: {:?}",
                    out.len(),
                    num_nan,
                    &out[..3.min(out.len())]
                );
            }
            None => {
                println!("Layer 3 failed!");
            }
        }

        // Continue with layers 4-27
        let mut hidden = result.expect("Layer 3 failed");
        for (layer_idx, layer) in model.layers.iter().skip(4).enumerate() {
            let actual_idx = layer_idx + 4;
            let result = model.forward_layer(
                layer,
                actual_idx,
                &hidden,
                &mut kv_cache,
                config.n_heads,
                config.head_dim,
                config.n_kv_heads,
            );

            match result {
                Some(out) => {
                    let num_nan: usize =
                        out.iter().filter(|&&v| v.is_nan()).count();
                    if num_nan > 0 {
                        println!("Layer {}: {} NaN", actual_idx, num_nan);
                        break;
                    }
                    hidden = out;
                }
                None => {
                    println!("Layer {} failed!", actual_idx);
                    break;
                }
            }
        }

        // Final norm and output projection
        println!("\n=== Final stage ===\n");
        let final_norm = model.norm.as_ref().expect("No final norm");
        let normed = model
            .apply_rms_norm(&hidden, final_norm, config.hidden_size)
            .expect("Final norm failed");
        let num_nan_normed: usize =
            normed.iter().filter(|&&v| v.is_nan()).count();
        println!("After final norm: {} NaN", num_nan_normed);

        let output = model.output.as_ref().expect("No output");
        println!(
            "Output weight: dtype={:?}, shape={:?}",
            output.dtype, output.shape
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("GPU support not available");
    }

    println!("\n=== Done ===");
}
