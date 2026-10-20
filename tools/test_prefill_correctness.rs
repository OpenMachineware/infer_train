// Test prefill correctness vs CPU reference
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::inference::InferenceContext;

fn main() {
    println!("=== Prefill Correctness Test ===\n");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/Qwen3-0.6B-Q5_K.gguf".to_string());

    println!("Loading model from: {}", model_path);

    let ctx = InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Gpu,
    )
    .expect("Failed to load model");

    println!("Model loaded: {} layers", ctx.model.layers.len());

    // Test with batch=4
    let batch = 4;
    let tokens: Vec<u32> = (1..=batch as u32).collect();

    // Run CPU forward for reference
    println!("\n=== CPU Reference ===");
    let mut cpu_kv = infer_train::inference::KVCache::new(
        ctx.model.config.n_kv_heads(),
        ctx.model.config.head_dim,
        512,
        ctx.model.layers.len(),
    );

    let mut cpu_outputs = Vec::new();
    for (i, &token) in tokens.iter().enumerate() {
        println!("Token {}: running CPU forward...", i);
        let output = ctx.model.forward_layer_cpu(
            token,
            i,
            &mut cpu_kv,
        ).expect("CPU forward failed");
        cpu_outputs.push(output);
        println!("  Output[0..4]: {:?}", &output[0..4]);
    }

    // Run GPU batched forward
    println!("\n=== GPU Batched ===");
    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");
    let mut gpu_kv = infer_train::inference::KVCacheGPU::new(
        gpu_ctx.device(),
        ctx.model.config.n_kv_heads(),
        ctx.model.config.head_dim,
        512,
        ctx.model.layers.len(),
    );

    // Create forward pool
    let reuse_pool = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut forward_pool = infer_train::buffer::ForwardBufferPool::new(
        gpu_ctx.device(),
        reuse_pool.clone(),
    );

    // Get embeddings
    let (embed_buffer, embed_batch) = ctx.model.embedding_lookup_batched_gpu(
        &tokens,
        gpu_ctx.device(),
    ).expect("Embedding lookup failed");

    println!("Embeddings created for {} tokens", embed_batch);

    // Run first layer
    let command_buffer = gpu_ctx.queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    let layer = &ctx.model.layers[0];
    let output = ctx.model.forward_layer_prefill_batched(
        layer,
        0,
        &embed_buffer,
        batch,
        &mut gpu_kv,
        gpu_ctx,
        &mut forward_pool,
        &encoder,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(output_buffer) = output {
        let output_ptr = output_buffer.contents() as *const f32;
        let output_len = output_buffer.length() as usize / 4;
        let gpu_output = unsafe {
            std::slice::from_raw_parts(output_ptr, output_len).to_vec()
        };

        println!("\nGPU output shape: {}", output_len);
        println!("GPU output[0..4]: {:?}", &gpu_output[0..4]);
        println!("CPU output[0..4]: {:?}", &cpu_outputs[0][0..4]);

        // Compare
        let diff: f32 = (0..4).map(|i| (gpu_output[i] - cpu_outputs[0][i]).abs()).sum();
        println!("Difference: {}", diff);
    }

    println!("\n=== Test Complete ===");
}
