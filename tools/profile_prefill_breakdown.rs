// Profile prefill to understand time breakdown
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::inference::{InferenceContext, KVCacheGPU};
use infer_train::buffer::ForwardBufferPool;
use std::rc::Rc;
use std::cell::RefCell;
use metal::Buffer;

fn main() {
    println!("=== Prefill Time Breakdown ===\n");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/Users/jia/Desktop/infer_train/DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string());

    println!("Loading model from: {}", model_path);

    let ctx = InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Gpu,
    ).expect("Failed to load model");

    println!("Model: {} layers, hidden={}, heads={}, head_dim={}",
        ctx.model.layers.len(),
        ctx.model.config.hidden_size,
        ctx.model.config.n_heads,
        ctx.model.config.head_dim,
    );

    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");
    let device = gpu_ctx.device().clone();
    let reuse_pool = Rc::new(RefCell::new(Vec::new()));

    // Test a single prompt length
    let prompt_len = 256;
    let tokens: Vec<u32> = (0..prompt_len as u32).collect();

    // Warmup run
    {
        let mut kv_cache = KVCacheGPU::new(&device, &ctx.model.config, 512);
        let (embed_buffer, batch) = ctx.model
            .embedding_lookup_batched_gpu(&tokens, &device)
            .expect("Embedding failed");
        let mut forward_pool = ForwardBufferPool::new(device.clone(), reuse_pool.clone());

        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let mut input_buffer: Option<Rc<Buffer>> = Some(embed_buffer.clone());
        for (layer_idx, layer) in ctx.model.layers.iter().enumerate() {
            if let Some(output) = ctx.model.forward_layer_prefill_batched(
                layer, layer_idx, input_buffer.as_ref().unwrap(), batch,
                &mut kv_cache, gpu_ctx, &mut forward_pool, &encoder,
            ) {
                input_buffer = Some(output);
            }
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    println!("\nRunning profiled forward pass for {} tokens...", prompt_len);

    // Profiled run - measure each operation
    let mut kv_cache = KVCacheGPU::new(&device, &ctx.model.config, 512);
    let (embed_buffer, batch) = ctx.model
        .embedding_lookup_batched_gpu(&tokens, &device)
        .expect("Embedding failed");

    let total_start = std::time::Instant::now();

    // Measure GEMM time separately
    let mut gemm_time = std::time::Duration::ZERO;
    let mut gemm_count = 0;

    // Run layer by layer with timing
    let mut layer_input: Option<Rc<Buffer>> = Some(embed_buffer.clone());
    for (layer_idx, layer) in ctx.model.layers.iter().enumerate() {
        let layer_start = std::time::Instant::now();

        let mut forward_pool = ForwardBufferPool::new(device.clone(), reuse_pool.clone());
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        if let Some(output) = ctx.model.forward_layer_prefill_batched(
            layer, layer_idx, layer_input.as_ref().unwrap(), batch,
            &mut kv_cache, gpu_ctx, &mut forward_pool, &encoder,
        ) {
            layer_input = Some(output);
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let layer_time = layer_start.elapsed();
        if layer_idx < 3 {
            println!("Layer {}: {:.1} ms", layer_idx, layer_time.as_secs_f64() * 1000.0);
        }

        // Approximate GEMM time (7 GEMMs per layer)
        // Each GEMM is a separate kernel dispatch
        gemm_count += 7;
    }

    let total_time = total_start.elapsed();

    println!("\n=== Summary ===");
    println!("Total time: {:.1} ms", total_time.as_secs_f64() * 1000.0);
    println!("Throughput: {:.1} tokens/s", prompt_len as f64 / total_time.as_secs_f64());
    println!("Per-layer average: {:.1} ms", total_time.as_secs_f64() * 1000.0 / 28.0);
    println!("Total GEMM ops: {}", gemm_count);

    // Estimate GEMM time based on known Flash Attention time
    // Flash Attention for 256 tokens: ~6.6ms per layer (from previous tests)
    let flash_attn_time = 6.6 * 28.0; // ms
    println!("\nEstimated Flash Attention: {:.1} ms ({:.1}%)",
        flash_attn_time, flash_attn_time / total_time.as_secs_f64() / 10.0);
    println!("Remaining (GEMM + other): {:.1} ms ({:.1}%)",
        total_time.as_secs_f64() * 1000.0 - flash_attn_time,
        (total_time.as_secs_f64() * 1000.0 - flash_attn_time) / total_time.as_secs_f64() / 10.0);
}
