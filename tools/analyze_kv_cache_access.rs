// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Analyze KV cache memory access during decode

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use metal::MTLResourceOptions;
        use std::rc::Rc;

        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load model");
        let gpu_ctx =
            MetalContext::new().expect("Failed to create Metal context");
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        let mut kv_cache = KVCacheGPU::new(
            &device,
            &ctx.config,
            ctx.config.max_position_embeddings,
        );

        // Prefill
        println!("=== Prefilling ===");
        let prefill_tokens: Vec<u32> = (0..100).collect();
        ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv_cache,
            &gpu_ctx,
        );

        println!("\n=== KV Cache Memory Access Analysis ===");

        let n_kv_heads = ctx.config.n_kv_heads;
        let head_dim = ctx.config.head_dim;
        let n_layers = ctx.model.layers.len();
        let seq_len = 101; // After prefill

        println!("n_kv_heads: {}, head_dim: {}", n_kv_heads, head_dim);
        println!("n_layers: {}", n_layers);
        println!("seq_len: {}", seq_len);

        // KV cache layout: FP16
        let kv_cache_per_layer = seq_len * n_kv_heads * head_dim * 2; // K + V, FP16 = 2 bytes
        let kv_cache_total = kv_cache_per_layer * n_layers * 2; // K + V

        println!("\nKV cache size:");
        println!(
            "  Per layer (K or V): {} bytes ({:.1} KB)",
            kv_cache_per_layer,
            kv_cache_per_layer as f64 / 1024.0
        );
        println!(
            "  Total (K+V, {} layers): {} bytes ({:.1} MB)",
            n_layers,
            kv_cache_total,
            kv_cache_total as f64 / 1024.0 / 1024.0
        );

        // Memory access per decode step
        // For each layer, Flash Attention reads:
        // - Q: n_heads * head_dim * 4 bytes (FP32)
        // - K cache: seq_len * n_kv_heads * head_dim * 2 bytes (FP16)
        // - V cache: seq_len * n_kv_heads * head_dim * 2 bytes (FP16)
        // - Writes output: n_heads * head_dim * 4 bytes

        let q_size = ctx.config.n_heads * head_dim * 4; // FP32
        let k_read = seq_len * n_kv_heads * head_dim * 2; // FP16
        let v_read = seq_len * n_kv_heads * head_dim * 2; // FP16
        let output_size = ctx.config.n_heads * head_dim * 4;

        let flash_attn_read_per_layer = q_size + k_read + v_read;
        let flash_attn_write_per_layer = output_size;

        println!("\nFlash Attention memory access per layer:");
        println!("  Q: {} bytes", q_size);
        println!(
            "  K cache read: {} bytes ({:.1} KB)",
            k_read,
            k_read as f64 / 1024.0
        );
        println!(
            "  V cache read: {} bytes ({:.1} KB)",
            v_read,
            v_read as f64 / 1024.0
        );
        println!("  Output: {} bytes", output_size);
        println!(
            "  Total read: {} bytes ({:.1} KB)",
            flash_attn_read_per_layer,
            flash_attn_read_per_layer as f64 / 1024.0
        );
        println!("  Total write: {} bytes", flash_attn_write_per_layer);

        // Total KV cache read per decode
        let total_kv_read = (k_read + v_read) * n_layers;
        println!("\nTotal KV cache read per decode (seq_len={}):", seq_len);
        println!(
            "  {} bytes ({:.2} MB)",
            total_kv_read,
            total_kv_read as f64 / 1024.0 / 1024.0
        );

        // Weight reads
        let layer = &ctx.model.layers[0];
        let per_layer_weight_bytes = {
            let wq = layer.wq.as_ref().expect("No wq");
            let wk = layer.wk.as_ref().expect("No wk");
            let wv = layer.wv.as_ref().expect("No wv");
            let wo = layer.wo.as_ref().expect("No wo");
            let gate = layer.w1.as_ref().expect("No gate");
            let up = layer.w3.as_ref().expect("No up");
            let down = layer.w2.as_ref().expect("No down");
            wq.data.as_bytes().len()
                + wk.data.as_bytes().len()
                + wv.data.as_bytes().len()
                + wo.data.as_bytes().len()
                + gate.data.as_bytes().len()
                + up.data.as_bytes().len()
                + down.data.as_bytes().len()
        };
        let total_weight_bytes = per_layer_weight_bytes * n_layers;

        println!("\nWeight reads per decode:");
        println!(
            "  {} bytes ({:.2} MB)",
            total_weight_bytes,
            total_weight_bytes as f64 / 1024.0 / 1024.0
        );

        // Total memory access
        let total_memory = total_weight_bytes + total_kv_read;
        println!("\n=== Total Memory Access ===");
        println!(
            "Weights: {:.2} MB",
            total_weight_bytes as f64 / 1024.0 / 1024.0
        );
        println!("KV cache: {:.2} MB", total_kv_read as f64 / 1024.0 / 1024.0);
        println!("Total: {:.2} MB", total_memory as f64 / 1024.0 / 1024.0);

        // Ratio
        let kv_ratio = total_kv_read as f64 / total_weight_bytes as f64;
        println!("\nKV cache / Weights ratio: {:.2}", kv_ratio);

        // Compare with llama.cpp
        // If llama.cpp has smaller KV cache reads, it might explain the bandwidth difference
        println!("\n=== Comparison ===");
        println!("Our bandwidth: 37.2 GB/s, time: 8.35 ms");
        println!("Effective memory: {:.2} MB", 37.2 * 8.35);
        println!(
            "Calculated memory: {:.2} MB",
            total_memory as f64 / 1024.0 / 1024.0
        );

        // Check if there's discrepancy
        let effective_mb = 37.2 * 8.35;
        let calculated_mb = total_memory as f64 / 1024.0 / 1024.0;
        println!(
            "Discrepancy: {:.2} MB ({:.1}%)",
            effective_mb - calculated_mb,
            (effective_mb / calculated_mb - 1.0) * 100.0
        );

        println!("\nllama.cpp bandwidth: 31 GB/s, time: 5.43 ms");
        println!("llama.cpp effective memory: {:.2} MB", 31.0 * 5.43);

        // The key question: why do we have more memory access?
        // Possible answers:
        // 1. Different KV cache format (we might be using FP32, llama.cpp uses FP16 or quantized)
        // 2. Different attention implementation
        // 3. Hidden intermediate buffers

        println!("\n=== Hypothesis ===");
        println!("If our bandwidth is higher but we're slower,");
        println!("we must be reading MORE data per decode step.");
        println!("");
        println!("Check:");
        println!("1. KV cache format - is it FP16 or quantized?");
        println!("2. Attention implementation - are we doing extra reads?");
        println!("3. Intermediate buffer reuse - are buffers being re-read?");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support");
    }
}
