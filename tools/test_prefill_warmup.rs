// Test prefill warmup effects
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::inference::{InferenceContext, KVCacheGPU};
use infer_train::buffer::ForwardBufferPool;
use std::rc::Rc;
use std::cell::RefCell;
use metal::Buffer;

fn main() {
    println!("=== Prefill Warmup Test ===\n");

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

    // Test a single prompt length multiple times
    let prompt_len = 256;
    let tokens: Vec<u32> = (0..prompt_len as u32).collect();

    println!("\nTesting {} tokens, 5 iterations:", prompt_len);
    println!("| Run | Time (ms) | Notes |");
    println!("|-----|-----------|--------|");

    for run in 1..=5 {
        // Reset KV cache for each run
        let mut kv_cache = KVCacheGPU::new(&device, &ctx.model.config, 512);

        // Get batched embedding
        let (embed_buffer, batch) = ctx.model
            .embedding_lookup_batched_gpu(&tokens, &device)
            .expect("Embedding failed");

        let mut forward_pool = ForwardBufferPool::new(device.clone(), reuse_pool.clone());

        // Batched forward pass - single command buffer
        let start = std::time::Instant::now();
        {
            let command_buffer = gpu_ctx.queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let mut input_buffer: Option<Rc<Buffer>> = Some(embed_buffer.clone());

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

        let notes = if run == 1 { "First run (warmup)" } else { "" };
        println!("| {:3} | {:9.1} | {} |", run, elapsed.as_secs_f64() * 1000.0, notes);
    }

    println!("\n=== Test Complete ===");
}
