// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Analyze decode kernel execution
//!
//! Count kernels and measure time spent in each category.

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::time::Instant;

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

        // Prefill
        let prefill_tokens: Vec<u32> = (0..100).collect();
        ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv_cache,
            &gpu_ctx,
        );

        println!("\n=== Decode Kernel Analysis ===");
        println!("Model: Qwen3-0.6B Q4_K");
        println!("Layers: {}", ctx.model.layers.len());

        // Count operations per decode
        let n_layers = ctx.model.layers.len();

        // Each layer has:
        // - 1 RMS Norm (attention_norm) - fused into FusedQKV
        // - 1 FusedQKV (3 GEMV in one GEMM)
        // - 2 RoPE (K and V)
        // - 1 Flash Attention
        // - 1 GEMV (wo)
        // - 1 FusedAddNorm (attention output + residual)
        // - 1 RMS Norm (ffn_norm) - fused into FusedGateUp
        // - 1 FusedGateUp (2 GEMV in one GEMM)
        // - 1 FusedSwiGLU
        // - 1 GEMV (down)
        // - 1 FusedAddNorm (FFN output + residual)

        println!("\n--- Operation Counts ---");
        println!("Per layer:");
        println!("  FusedQKV:         1 (replaces 3 GEMV + 1 norm)");
        println!("  RoPE:             2");
        println!("  Flash Attention:  1");
        println!("  GEMV (wo):        1");
        println!("  FusedAddNorm:     2");
        println!("  FusedGateUp:      1 (replaces 2 GEMV + 1 norm)");
        println!("  FusedSwiGLU:      1");
        println!("  GEMV (down):      1");
        println!("  Total: {} ops/layer", 10);

        let total_ops = 10 * n_layers;
        println!("\nTotal for {} layers: {} operations", n_layers, total_ops);

        // GEMV count: wo + down = 2 per layer
        let total_gemv = 2 * n_layers;
        println!("GEMV ops: {}", total_gemv);

        // Measure actual decode
        println!("\n--- Timing Breakdown ---");

        // Time individual operations
        let iterations = 20;

        // GEMV (wo projection)
        let layer = &ctx.model.layers[0];
        let hidden_size = ctx.config.hidden_size;
        let wo = layer.wo.as_ref().expect("No wo");
        let wo_out_size = wo.shape[1];

        use metal::MTLResourceOptions;
        use std::rc::Rc;

        let input_buffer: Rc<metal::Buffer> =
            Rc::new(device.new_buffer_with_data(
                vec![1.0f32; hidden_size].as_ptr() as *const _,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        let out_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (wo_out_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let queue = gpu_ctx.queue();

        // Test GEMV time
        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                wo,
                wo_out_size,
                &gpu_ctx,
                &out_buffer,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let gemv_time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("GEMV (wo, K=1024): {:.2} µs", gemv_time);

        // Test down projection
        let down = layer.w2.as_ref().expect("No down");
        let down_out_size = hidden_size;
        let intermediate_size = ctx.config.intermediate_size;
        let down_input: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (intermediate_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let down_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (down_out_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &down_input,
                intermediate_size,
                down,
                down_out_size,
                &gpu_ctx,
                &down_out,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let gemv_down_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("GEMV (down, K=3072): {:.2} µs", gemv_down_time);

        // Test RMS Norm
        let norm = layer.attention_norm.as_ref().expect("No norm");
        let norm_buffer = gpu_ctx.get_or_create_buffer(norm);
        let norm_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
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
                &norm_out,
                1,
                hidden_size,
                1e-5,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let norm_time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("RMS Norm: {:.2} µs", norm_time);

        // Test RoPE
        let n_heads = ctx.config.n_heads;
        let head_dim = ctx.config.head_dim;
        let rope_in: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (n_heads * head_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let rope_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (n_heads * head_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();
            gpu_ctx.rope_neox_encode(
                &encoder,
                &rope_in,
                &[100i32],
                1,
                n_heads,
                head_dim,
                10000.0,
                1.0,
                &rope_out,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let rope_time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("RoPE: {:.2} µs", rope_time);

        // Test FusedSwiGLU
        let silu_in: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (intermediate_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let up_in: Rc<metal::Buffer> = Rc::new(device.new_buffer(
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
                &silu_in,
                &up_in,
                &swiglu_out,
                intermediate_size,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let swiglu_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("FusedSwiGLU: {:.2} µs", swiglu_time);

        // Test FusedAddNorm
        let lhs: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let rhs: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let add_norm_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();
            gpu_ctx.fused_add_norm_encode(
                &encoder,
                &lhs,
                &rhs,
                &norm_buffer,
                &add_norm_out,
                hidden_size,
                1e-5,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let add_norm_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("FusedAddNorm: {:.2} µs", add_norm_time);

        // Calculate totals
        println!("\n--- Estimated Total ---");
        let gemv_total = (gemv_time + gemv_down_time) * total_gemv as f64 / 2.0;
        let rope_total = rope_time * 2.0 * n_layers as f64;
        let swiglu_total = swiglu_time * n_layers as f64;
        let add_norm_total = add_norm_time * 2.0 * n_layers as f64;
        let fa_total = 8.5 * n_layers as f64; // From Flash Attention profiler

        let estimated =
            gemv_total + rope_total + swiglu_total + add_norm_total + fa_total;

        println!("GEMV total:       {:.1} µs", gemv_total);
        println!("RoPE total:       {:.1} µs", rope_total);
        println!("FusedSwiGLU total:{:.1} µs", swiglu_total);
        println!("FusedAddNorm total:{:.1} µs", add_norm_total);
        println!("Flash Attn total: {:.1} µs", fa_total);
        println!("Estimated:        {:.1} µs", estimated);

        // Compare with actual
        println!("\n--- Actual Decode ---");
        let start = Instant::now();
        for _ in 0..10 {
            ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);
        }
        let actual = start.elapsed().as_micros() as f64 / 10.0;
        println!("Actual:           {:.1} µs", actual);

        let unaccounted = actual - estimated;
        println!(
            "\nUnaccounted:      {:.1} µs ({:.1}%)",
            unaccounted,
            unaccounted / actual * 100.0
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
