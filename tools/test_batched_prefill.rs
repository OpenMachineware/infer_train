// Test batched prefill end-to-end
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::inference::InferenceContext;

fn main() {
    println!("=== Batched Prefill Test ===\n");

    // Load model
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/Qwen3-0.6B-Q5_K.gguf".to_string());

    println!("Loading model from: {}", model_path);

    let ctx = InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Gpu,
    )
    .expect("Failed to load model");

    println!("Model loaded: {} layers", ctx.model.layers.len());
    println!("Hidden size: {}", ctx.model.config.hidden_size);
    println!("Heads: {}", ctx.model.config.n_heads);
    println!("Head dim: {}", ctx.model.config.head_dim);

    // Get Metal context
    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");

    // Test batch sizes
    let batch_sizes = [1, 4, 16, 32];

    for batch in batch_sizes.iter() {
        println!("\n--- Batch size: {} ---", batch);

        // Create test input (batch tokens)
        let tokens: Vec<u32> = (0..*batch as u32).collect();
        let hidden_size = ctx.model.config.hidden_size;

        // Test batched embedding lookup
        let start = std::time::Instant::now();
        let (embed_buffer, embed_batch) = ctx
            .model
            .embedding_lookup_batched_gpu(&tokens, gpu_ctx.device())
            .expect("Embedding lookup failed");
        let embed_time = start.elapsed();

        assert_eq!(embed_batch, *batch);
        println!("  Embedding lookup: {:?}", embed_time);
        println!("  Embedding buffer size: {} bytes", embed_buffer.length());

        // Test batched RMSNorm
        let norm_weight = ctx.model.layers[0]
            .attention_norm
            .as_ref()
            .expect("No norm weight");
        let norm_weight_buffer = gpu_ctx.get_or_create_buffer(norm_weight);

        let normed_buffer = std::rc::Rc::new(gpu_ctx.device().new_buffer(
            (batch * hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let start = std::time::Instant::now();
        {
            let command_buffer = gpu_ctx.queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            gpu_ctx.rms_norm_encode(
                &encoder,
                &embed_buffer,
                &norm_weight_buffer,
                &normed_buffer,
                *batch,
                hidden_size,
                1e-5,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        let norm_time = start.elapsed();
        println!("  RMSNorm: {:?}", norm_time);

        // Test F32->F16 conversion
        let f16_buffer = std::rc::Rc::new(gpu_ctx.device().new_buffer(
            (batch * hidden_size * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let start = std::time::Instant::now();
        {
            let command_buffer = gpu_ctx.queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            gpu_ctx.f32_to_f16_encode(
                &encoder,
                &normed_buffer,
                &f16_buffer,
                batch * hidden_size,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        let conv_time = start.elapsed();
        println!("  F32->F16 conversion: {:?}", conv_time);

        // Test GEMM for Q projection
        let layer = &ctx.model.layers[0];
        let wq = layer.wq.as_ref().expect("No wq");
        let wq_buffer = gpu_ctx.get_or_create_buffer(wq);
        let q_out_size = wq.shape[1];

        let q_buffer = std::rc::Rc::new(gpu_ctx.device().new_buffer(
            (batch * q_out_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let start = std::time::Instant::now();
        {
            let command_buffer = gpu_ctx.queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            gpu_ctx.mul_mm_q5_k_encode(
                &encoder,
                &wq_buffer,
                &f16_buffer,
                &q_buffer,
                q_out_size,
                *batch,
                hidden_size,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        let gemm_time = start.elapsed();
        println!("  Q projection GEMM: {:?}", gemm_time);

        // Check output for NaN
        let q_ptr = q_buffer.contents() as *const f32;
        let q_data = unsafe {
            std::slice::from_raw_parts(q_ptr, batch * q_out_size).to_vec()
        };

        let has_nan = q_data.iter().any(|&x| x.is_nan());
        let has_inf = q_data.iter().any(|&x| x.is_infinite());
        let min = q_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = q_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = q_data.iter().sum();

        println!("  Q output stats:");
        println!("    NaN: {}, Inf: {}", has_nan, has_inf);
        println!("    Min: {:.6}, Max: {:.6}", min, max);
        println!("    Sum: {:.2}", sum);

        if has_nan || has_inf {
            println!("  ❌ ERROR: Invalid output!");
        } else {
            println!("  ✓ Output is valid");
        }
    }

    println!("\n=== Test Complete ===");
}
