// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Detailed decode operation profiling
//!
//! Measures time spent in each operation category during decode.

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use metal::MTLResourceOptions;
        use std::rc::Rc;
        use std::time::Instant;

        // Load model
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load model");
        let gpu_ctx =
            MetalContext::new().expect("Failed to create Metal context");
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        // Create KV cache
        let mut kv_cache = KVCacheGPU::new(
            &device,
            &ctx.config,
            ctx.config.max_position_embeddings,
        );

        // Prefill
        let prefill_tokens: Vec<u32> = (0..100).collect();
        ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv_cache,
            &gpu_ctx,
        );

        println!("\n=== Detailed Decode Operation Profiling ===");
        println!("Layers: {}", ctx.model.layers.len());

        let layer = &ctx.model.layers[0];
        let hidden_size = ctx.config.hidden_size;
        let n_heads = ctx.config.n_heads;
        let head_dim = ctx.config.head_dim;

        // Create test buffers
        let input_buffer: Rc<metal::Buffer> =
            Rc::new(device.new_buffer_with_data(
                vec![1.0f32; hidden_size].as_ptr() as *const _,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

        let iterations = 10;

        // Test GEMV (Q projection)
        println!("\n--- GEMV Operations ---");
        let wq = layer.wq.as_ref().expect("No wq");
        let q_out_size = wq.shape[1];
        let out_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (q_out_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();

            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                wq,
                q_out_size,
                &gpu_ctx,
                &out_buffer,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let gemv_q_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("Q projection (M=2048, K=1024): {:.2} µs", gemv_q_time);

        // Test Gate projection
        let gate_proj = layer.w1.as_ref().expect("No gate");
        let intermediate_size = gate_proj.shape[1];

        let gate_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (intermediate_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();

            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                gate_proj,
                intermediate_size,
                &gpu_ctx,
                &gate_buffer,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let gemv_gate_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("Gate projection (M=3072, K=1024): {:.2} µs", gemv_gate_time);

        // Test Down projection
        let down_proj = layer.w2.as_ref().expect("No down");
        let down_out_size = hidden_size;
        let down_input_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (intermediate_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let down_out_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (down_out_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();

            ctx.model.gemv_encode(
                &encoder,
                &down_input_buffer,
                intermediate_size,
                down_proj,
                down_out_size,
                &gpu_ctx,
                &down_out_buffer,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let gemv_down_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("Down projection (M=1024, K=3072): {:.2} µs", gemv_down_time);

        // Test RMS Norm
        println!("\n--- RMS Norm ---");
        let norm = layer.attention_norm.as_ref().expect("No norm");
        let norm_buffer = gpu_ctx.get_or_create_buffer(norm);
        let norm_out_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();

            gpu_ctx.rms_norm_encode(
                &encoder,
                &input_buffer,
                &norm_buffer,
                &norm_out_buffer,
                1,
                hidden_size,
                1e-5,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let norm_time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("RMS Norm (size=1024): {:.2} µs", norm_time);

        // Test FusedSwiGLU
        println!("\n--- FusedSwiGLU ---");
        let silu_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (intermediate_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let up_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (intermediate_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let swiglu_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (intermediate_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();

            gpu_ctx.fused_swiglu_encode(
                &encoder,
                &silu_out,
                &up_out,
                &swiglu_out,
                intermediate_size,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let swiglu_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("FusedSwiGLU (size=3072): {:.2} µs", swiglu_time);

        // Test RoPE
        println!("\n--- RoPE ---");
        let q_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (n_heads * head_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let rope_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (n_heads * head_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();

            gpu_ctx.rope_neox_encode(
                &encoder,
                &q_buffer,
                &[100i32],
                1,
                n_heads,
                head_dim,
                10000.0,
                1.0,
                &rope_buffer,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let rope_time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("RoPE (n_heads=16, head_dim=128): {:.2} µs", rope_time);

        // Summary
        println!("\n=== Operation Summary ===");
        println!("GEMV Q:          {:.2} µs", gemv_q_time);
        println!("GEMV Gate:       {:.2} µs", gemv_gate_time);
        println!("GEMV Down:       {:.2} µs", gemv_down_time);
        println!("RMS Norm:        {:.2} µs", norm_time);
        println!("FusedSwiGLU:     {:.2} µs", swiglu_time);
        println!("RoPE:            {:.2} µs", rope_time);

        // Estimate per-layer breakdown (excluding Flash Attention)
        let n_layers = ctx.model.layers.len();

        // Each layer:
        // - 1 RMS Norm (attention_norm)
        // - 3 GEMV (Q/K/V) - FusedQKV
        // - 2 RoPE (K and V)
        // - 1 Flash Attention
        // - 1 GEMV (wo)
        // - 1 FusedAddNorm (after attention)
        // - 1 FusedGateUp (gate + up in one GEMM)
        // - 1 FusedSwiGLU
        // - 1 GEMV (down)
        // - 1 FusedAddNorm (after FFN)

        let attention_no_fa =
            norm_time + gemv_q_time * 3.0 + rope_time * 2.0 + gemv_q_time; // norm + QKV + RoPE + wo
        let ffn_no_fa =
            norm_time + gemv_gate_time * 2.0 + swiglu_time + gemv_down_time; // norm + gate+up + swiglu + down

        let per_layer_est = attention_no_fa + ffn_no_fa;
        let total_est = per_layer_est * n_layers as f64;
        let est_tps = 1_000_000.0 / total_est;

        println!("\n=== Estimated Decode (excluding Flash Attention) ===");
        println!("Attention/layer (no FA): {:.2} µs", attention_no_fa);
        println!("FFN/layer:               {:.2} µs", ffn_no_fa);
        println!("Total (no FA):           {:.2} µs", total_est);
        println!("Est t/s (no FA):         {:.1}", est_tps);

        // Compare with actual
        println!("\n=== Actual Decode ===");
        let start = Instant::now();
        for _ in 0..10 {
            ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);
        }
        let actual_time = start.elapsed().as_micros() as f64 / 10.0;
        let actual_tps = 1_000_000.0 / actual_time;

        println!("Actual time:      {:.2} µs", actual_time);
        println!("Actual t/s:       {:.1}", actual_tps);

        // Flash Attention is the difference
        let fa_time = (actual_time - total_est) / n_layers as f64;
        println!("\n=== Imputed Flash Attention Time ===");
        println!("FA/layer:         {:.2} µs", fa_time);
        println!("FA total:         {:.2} µs", fa_time * n_layers as f64);
        println!(
            "FA %% of total:    {:.1}%%",
            (fa_time * n_layers as f64) / actual_time * 100.0
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
