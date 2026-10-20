// Benchmark decode-style prefill (token by token)
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::inference::InferenceContext;

fn main() {
    println!("=== Decode-Style Prefill Performance ===\n");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/Qwen3-0.6B-Q5_K.gguf".to_string());

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
        ctx.model.config.n_heads(),
        ctx.model.config.head_dim
    );

    // Test different prompt lengths
    let prompt_lengths = [16, 32, 64, 128, 256];

    println!("\n| Prompt | Time (ms) | Throughput (t/s) |");
    println!("|--------|-----------|-------------------|");

    for prompt_len in prompt_lengths.iter() {
        // Create KV cache
        let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");
        let mut kv_cache = infer_train::inference::KVCacheGPU::new(
            gpu_ctx.device(),
            ctx.model.config.n_kv_heads(),
            ctx.model.config.head_dim,
            512,
            ctx.model.layers.len(),
        );

        let start = std::time::Instant::now();

        // Process tokens one by one (decode style)
        for pos in 0..*prompt_len {
            let token = (pos % 100) as u32 + 1;

            // Run one forward pass
            let command_buffer = gpu_ctx.queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Use decode-style forward
            ctx.model.forward_layer_gpu(
                token,
                pos,
                &mut kv_cache,
                gpu_ctx,
                &encoder,
            );

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
