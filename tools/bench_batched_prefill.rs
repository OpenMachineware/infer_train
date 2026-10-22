// Benchmark batched prefill performance
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::buffer::ForwardBufferPool;
use infer_train::inference::{InferenceContext, KVCacheGPU};
use metal::Buffer;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("=== Batched Prefill Performance ===\n");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/Users/jia/Desktop/infer_train/DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string());

    println!("Loading model from: {}", model_path);

    let ctx = InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Gpu,
    )
    .expect("Failed to load model");

    println!(
        "Model: {} layers, hidden={}, heads={}, head_dim={}",
        ctx.model.layers.len(),
        ctx.model.config.hidden_size,
        ctx.model.config.n_heads,
        ctx.model.config.head_dim,
    );

    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");
    let device = gpu_ctx.device().clone();
    let reuse_pool = Rc::new(RefCell::new(Vec::new()));

    // Test prompt lengths
    let prompt_lengths = [16, 32, 64, 128, 256];

    println!("\n| Prompt | Time (ms) | Throughput (t/s) |");
    println!("|--------|-----------|-------------------|");

    for prompt_len in prompt_lengths.iter() {
        // Create test tokens
        let tokens: Vec<u32> = (0..*prompt_len as u32).collect();

        // Reset KV cache for each run
        let mut kv_cache = KVCacheGPU::new(&device, &ctx.model.config, 512);

        let mut forward_pool =
            ForwardBufferPool::new(device.clone(), reuse_pool.clone());

        // Batched forward pass - single command buffer
        // Include embedding lookup in timing!
        let start = std::time::Instant::now();

        // Get batched embedding (CPU-side operation)
        let (embed_buffer, batch) = ctx
            .model
            .embedding_lookup_batched_gpu(&tokens, &device)
            .expect("Embedding failed");

        {
            let command_buffer = gpu_ctx.queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let mut input_buffer: Option<Rc<Buffer>> =
                Some(embed_buffer.clone());

            for (layer_idx, layer) in ctx.model.layers.iter().enumerate() {
                if let Some(output) = ctx.model.forward_layer_prefill_batched(
                    layer,
                    layer_idx,
                    input_buffer.as_ref().unwrap(),
                    batch,
                    &mut kv_cache,
                    gpu_ctx,
                    &mut forward_pool,
                    &encoder,
                ) {
                    input_buffer = Some(output);
                }
            }

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        let elapsed = start.elapsed();
        let tps = *prompt_len as f64 / elapsed.as_secs_f64();

        println!(
            "| {:6} | {:9.1} | {:17.1} |",
            prompt_len,
            elapsed.as_secs_f64() * 1000.0,
            tps
        );
    }

    println!("\n=== Benchmark Complete ===");
}
