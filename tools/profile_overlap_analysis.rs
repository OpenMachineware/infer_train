// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Find operations that prevent kernel overlap
//!
//! Test different operation combinations to see which ones can overlap.

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
        let prefill_tokens: Vec<u32> = (0..100).collect();
        ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv_cache,
            &gpu_ctx,
        );

        println!("\n=== Kernel Overlap Analysis ===");

        let layer = &ctx.model.layers[0];
        let hidden_size = ctx.config.hidden_size;
        let intermediate_size = ctx.config.intermediate_size;
        let n_heads = ctx.config.n_heads;
        let head_dim = ctx.config.head_dim;

        let wo = layer.wo.as_ref().expect("No wo");
        let wo_out_size = wo.shape[1];
        let down = layer.w2.as_ref().expect("No down");

        let iterations = 10;

        // Create buffers
        let input_buffer: Rc<metal::Buffer> =
            Rc::new(device.new_buffer_with_data(
                vec![1.0f32; hidden_size].as_ptr() as *const _,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        let wo_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (wo_out_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let down_in: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (intermediate_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let down_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        // Test 1: GEMV + GEMV (same input, different weights)
        println!("\n--- Test 1: Two GEMV with same input ---");
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
                &wo_out,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &down_in,
                intermediate_size,
                down,
                hidden_size,
                &gpu_ctx,
                &down_out,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let sequential = start.elapsed().as_micros() as f64 / iterations as f64;

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
                &wo_out,
            );
            encoder.end_encoding();
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &down_in,
                intermediate_size,
                down,
                hidden_size,
                &gpu_ctx,
                &down_out,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let batched = start.elapsed().as_micros() as f64 / iterations as f64;

        println!("Sequential: {:.1} µs", sequential);
        println!("Batched:    {:.1} µs", batched);
        println!("Speedup:    {:.2}x", sequential / batched);

        // Test 2: GEMV + RoPE (dependent - RoPE uses GEMV output)
        println!("\n--- Test 2: GEMV -> RoPE (data dependency) ---");
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
            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                wo,
                wo_out_size,
                &gpu_ctx,
                &rope_in,
            );
            encoder.end_encoding();
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
        let gemv_rope = start.elapsed().as_micros() as f64 / iterations as f64;

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
                &wo_out,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
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
        let gemv_then_rope =
            start.elapsed().as_micros() as f64 / iterations as f64;

        println!("GEMV+RoPE (same CB): {:.1} µs", gemv_rope);
        println!("GEMV then RoPE:      {:.1} µs", gemv_then_rope);
        println!(
            "Overhead:            {:.1} µs ({:.1}%)",
            gemv_then_rope - gemv_rope,
            (gemv_then_rope - gemv_rope) / gemv_then_rope * 100.0
        );

        // Test 3: Multiple layers worth of operations
        println!("\n--- Test 3: Full layer (7 operations) ---");
        let norm = layer.attention_norm.as_ref().expect("No norm");
        let norm_buffer = gpu_ctx.get_or_create_buffer(norm);
        let norm_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let q_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (n_heads * head_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let k_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (8 * head_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let v_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (8 * head_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let wq = layer.wq.as_ref().expect("No wq");
        let wk = layer.wk.as_ref().expect("No wk");
        let wv = layer.wv.as_ref().expect("No wv");
        let q_size = wq.shape[1];
        let kv_size = wk.shape[1];

        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();

            // Norm
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

            // Q projection
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &norm_out,
                hidden_size,
                wq,
                q_size,
                &gpu_ctx,
                &q_out,
            );
            encoder.end_encoding();

            // K projection
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &norm_out,
                hidden_size,
                wk,
                kv_size,
                &gpu_ctx,
                &k_out,
            );
            encoder.end_encoding();

            // V projection
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &norm_out,
                hidden_size,
                wv,
                kv_size,
                &gpu_ctx,
                &v_out,
            );
            encoder.end_encoding();

            cb.commit();
            cb.wait_until_completed();
        }
        let layer_batched =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("4 ops batched: {:.1} µs", layer_batched);

        // Compare with separate CBs
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

            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &norm_out,
                hidden_size,
                wq,
                q_size,
                &gpu_ctx,
                &q_out,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &norm_out,
                hidden_size,
                wk,
                kv_size,
                &gpu_ctx,
                &k_out,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &norm_out,
                hidden_size,
                wv,
                kv_size,
                &gpu_ctx,
                &v_out,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let layer_sequential =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("4 ops sequential: {:.1} µs", layer_sequential);
        println!("Speedup: {:.2}x", layer_sequential / layer_batched);

        // Compare with actual decode
        println!("\n=== Comparison ===");
        let start = Instant::now();
        for _ in 0..10 {
            ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);
        }
        let actual = start.elapsed().as_micros() as f64 / 10.0;
        println!("Full decode: {:.1} µs", actual);
        println!("4 ops (single layer): {:.1} µs", layer_batched);
        println!("Ratio: {:.1}x", actual / layer_batched);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
