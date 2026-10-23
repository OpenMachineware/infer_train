// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Analyze weight memory access patterns during decode

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load model");
        let gpu_ctx =
            MetalContext::new().expect("Failed to create Metal context");
        let device = gpu_ctx.device();

        let mut kv_cache = KVCacheGPU::new(
            &device,
            &ctx.config,
            ctx.config.max_position_embeddings,
        );

        println!("=== Weight Access Pattern Analysis ===\n");

        // Prefill
        println!("Prefilling...");
        let prefill_tokens: Vec<u32> = (0..100).collect();
        ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv_cache,
            &gpu_ctx,
        );

        // Calculate total weight memory
        let n_layers = ctx.model.layers.len();
        let mut total_weight_bytes = 0usize;
        let mut weight_sizes: Vec<(String, usize)> = Vec::new();

        for (i, layer) in ctx.model.layers.iter().enumerate() {
            let mut layer_weight = 0usize;

            if let Some(wq) = &layer.wq {
                let size = wq.data.as_bytes().len();
                layer_weight += size;
                if i == 0 {
                    weight_sizes.push(("wq".to_string(), size));
                }
            }
            if let Some(wk) = &layer.wk {
                let size = wk.data.as_bytes().len();
                layer_weight += size;
                if i == 0 {
                    weight_sizes.push(("wk".to_string(), size));
                }
            }
            if let Some(wv) = &layer.wv {
                let size = wv.data.as_bytes().len();
                layer_weight += size;
                if i == 0 {
                    weight_sizes.push(("wv".to_string(), size));
                }
            }
            if let Some(wo) = &layer.wo {
                let size = wo.data.as_bytes().len();
                layer_weight += size;
                if i == 0 {
                    weight_sizes.push(("wo".to_string(), size));
                }
            }
            if let Some(w1) = &layer.w1 {
                let size = w1.data.as_bytes().len();
                layer_weight += size;
                if i == 0 {
                    weight_sizes.push(("w1 (gate)".to_string(), size));
                }
            }
            if let Some(w3) = &layer.w3 {
                let size = w3.data.as_bytes().len();
                layer_weight += size;
                if i == 0 {
                    weight_sizes.push(("w3 (up)".to_string(), size));
                }
            }
            if let Some(w2) = &layer.w2 {
                let size = w2.data.as_bytes().len();
                layer_weight += size;
                if i == 0 {
                    weight_sizes.push(("w2 (down)".to_string(), size));
                }
            }

            total_weight_bytes += layer_weight;
        }

        println!("\n=== Layer 0 Weight Sizes ===");
        for (name, size) in &weight_sizes {
            println!("  {}: {:.2} MB", name, *size as f64 / 1024.0 / 1024.0);
        }

        println!("\n=== Total Weights ===");
        println!(
            "Total weight memory: {:.2} MB",
            total_weight_bytes as f64 / 1024.0 / 1024.0
        );
        println!(
            "Per layer: {:.2} MB",
            total_weight_bytes as f64 / 1024.0 / 1024.0 / n_layers as f64
        );

        // Calculate memory bandwidth requirements
        println!("\n=== Memory Bandwidth Requirements ===");

        let decode_time_ms = 8.7; // From benchmark
        let bandwidth_gbs =
            total_weight_bytes as f64 / (decode_time_ms / 1000.0) / 1e9;
        println!("Decode time: {} ms", decode_time_ms);
        println!("Required bandwidth: {:.1} GB/s", bandwidth_gbs);
        println!(
            "Utilization (100 GB/s peak): {:.1}%",
            bandwidth_gbs / 100.0 * 100.0
        );

        // Cache analysis
        println!("\n=== GPU Cache Analysis ===");
        println!("Apple Silicon GPU caches:");
        println!("  - L2 cache: ~3-4 MB (shared across GPU)");
        println!("  - System cache: ~12-24 MB (shared with CPU)");
        println!("  - Total cache: ~15-28 MB");
        println!(
            "\nModel weights: {:.2} MB",
            total_weight_bytes as f64 / 1024.0 / 1024.0
        );
        println!(
            "Cache coverage: {:.1}%",
            (28.0 * 1024.0 * 1024.0 / total_weight_bytes as f64) * 100.0
        );

        println!("\n=== Access Pattern ===");
        println!("Decode sequence (batch=1):");
        println!("  1. Read embedding: ~4 KB");
        println!("  2. For each layer:");
        println!("     - Read norm weights: ~4 KB");
        println!("     - Read Q/K/V/O weights: ~5-8 MB");
        println!("     - Read FFN weights: ~5-8 MB");
        println!("     - Read/write KV cache: ~few KB");
        println!("  3. Read output layer weights: ~600 MB (for 0.6B model)");
        println!("  4. Read final norm weights: ~4 KB");

        println!("\nKey insight:");
        println!(
            "  - Total weight reads: {:.2} MB",
            total_weight_bytes as f64 / 1024.0 / 1024.0
        );
        println!("  - Cache can hold: ~28 MB");
        println!("  - Each layer: ~12-16 MB weights");
        println!("  - Cache can hold ~2 layers worth of weights");
        println!("  - But layers are sequential, so cache won't help much");

        println!("\n=== Comparison with llama.cpp ===");
        println!("Hypothesis: llama.cpp may have:");
        println!("  1. More efficient weight packing");
        println!("  2. Better memory access coalescing");
        println!("  3. Fused operations reducing intermediate memory");
        println!("  4. Optimized kernel scheduling");

        println!("\n=== Next Investigation ===");
        println!("Focus on:");
        println!("  1. Are our weights stored contiguously?");
        println!("  2. Are there unnecessary memory copies?");
        println!("  3. What is the actual dispatch count?");
        println!("  4. What is the encoder overhead?");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support");
    }
}
