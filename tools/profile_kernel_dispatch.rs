// Measure total kernel dispatch count and overhead
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::inference::{InferenceContext, KVCacheGPU};
use infer_train::buffer::ForwardBufferPool;
use std::rc::Rc;
use std::cell::RefCell;
use metal::Buffer;

fn main() {
    println!("=== Kernel Dispatch Analysis ===\n");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/Users/jia/Desktop/infer_train/DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string());

    let ctx = InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Gpu,
    ).expect("Failed to load model");

    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");
    let device = gpu_ctx.device().clone();
    let reuse_pool = Rc::new(RefCell::new(Vec::new()));

    println!("Model: {} layers", ctx.model.layers.len());

    // Single batch size for analysis
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

    // Measure with GPU timing
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

    // Measure GPU execution time
    let start = std::time::Instant::now();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let cpu_time = start.elapsed();

    let total_time_ms = cpu_time.as_secs_f64() * 1000.0;

    println!("\n=== Results for {} tokens ===", prompt_len);
    println!("Total execution time: {:.1} ms", total_time_ms);

    // Estimate operations
    let layers = ctx.model.layers.len();
    let kernels_per_layer = 24; // Approximate
    let total_kernels = layers * kernels_per_layer;
    println!("\nEstimated kernel dispatches: {} ({} per layer)", total_kernels, kernels_per_layer);
    println!("Average time per kernel: {:.3} ms", total_time_ms / total_kernels as f64);

    // GEMM analysis
    let gemms_per_layer = 7;
    let total_gemms = layers * gemms_per_layer;
    println!("\nGEMM dispatches: {} ({} per layer)", total_gemms, gemms_per_layer);

    // Estimate GEMM time based on single GEMM benchmarks
    let gemm_time_estimate_ms = 0.73; // From Q projection benchmark for batch=256
    let total_gemm_estimate = total_gemms as f64 * gemm_time_estimate_ms;
    println!("Estimated GEMM time: {:.1} ms", total_gemm_estimate);
    println!("GEMM % of total: {:.1}%", total_gemm_estimate / total_time_ms * 100.0);
}
