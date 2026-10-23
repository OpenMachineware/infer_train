// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Operation-level timing profiler
//!
//! Measures time spent in each operation type during decode.

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use metal::MTLResourceOptions;
        use std::collections::HashMap;
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

        // Warmup
        let warmup_tokens: Vec<u32> = (0..10).collect();
        ctx.model.forward_gpu_native_batched(
            &warmup_tokens,
            &mut kv_cache,
            &gpu_ctx,
        );

        // Reset for profiling
        kv_cache.clear();

        // Prefill
        let prefill_tokens: Vec<u32> = (0..100).collect();
        ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv_cache,
            &gpu_ctx,
        );

        println!("\n=== Operation-Level Timing ===");

        // Profile each operation type separately
        let mut timings: HashMap<&str, f64> = HashMap::new();
        let iterations = 10;

        // Test GEMV (Q projection)
        println!("\n--- GEMV (Q projection) ---");
        let layer = &ctx.model.layers[0];
        let wq = layer.wq.as_ref().expect("No wq");
        let hidden_size = ctx.config.hidden_size;
        let q_out_size = wq.shape[1];

        let input_buffer: Rc<metal::Buffer> =
            Rc::new(device.new_buffer_with_data(
                vec![1.0f32; hidden_size].as_ptr() as *const _,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        let q_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
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
                &q_buffer,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let gemv_time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("Time: {:.2} µs", gemv_time);
        timings.insert("GEMV", gemv_time);

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
                1, // batch
                hidden_size,
                1e-5,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let norm_time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("Time: {:.2} µs", norm_time);
        timings.insert("RMS Norm", norm_time);

        // Calculate total decode time estimate
        let n_layers = ctx.model.layers.len();
        println!("\n--- Estimated Decode Time ---");
        println!("Layers: {}", n_layers);

        // Each layer has:
        // - 1 RMS Norm (attention)
        // - 1 FusedQKV (3 GEMV)
        // - 1 Flash Attention
        // - 1 GEMV (wo)
        // - 1 FusedAddNorm
        // - 1 FusedGateUp (2 GEMV)
        // - 1 FusedSwiGLU
        // - 1 GEMV (down)
        // - 1 FusedAddNorm

        // Rough estimate:
        let gemv_per_layer = 7; // Q, K, V, wo, gate, up, down
        let norm_per_layer = 2; // attention_norm, ffn_norm

        let total_gemv = gemv_per_layer as f64 * n_layers as f64 * gemv_time;
        let total_norm = norm_per_layer as f64 * n_layers as f64 * norm_time;

        println!(
            "Total GEMV time: {:.2} µs ({} ops)",
            total_gemv,
            gemv_per_layer * n_layers
        );
        println!(
            "Total Norm time: {:.2} µs ({} ops)",
            total_norm,
            norm_per_layer * n_layers
        );
        println!("Estimated total: {:.2} µs", total_gemv + total_norm);

        let tokens_per_sec = 1_000_000.0 / (total_gemv + total_norm);
        println!("Estimated throughput: {:.1} t/s", tokens_per_sec);

        // Compare with actual
        println!("\n--- Actual Decode Performance ---");
        let start = Instant::now();
        let test_tokens = 10;
        for _ in 0..test_tokens {
            ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);
        }
        let actual_time =
            start.elapsed().as_micros() as f64 / test_tokens as f64;
        let actual_tps = 1_000_000.0 / actual_time;

        println!("Actual time: {:.2} µs", actual_time);
        println!("Actual throughput: {:.1} t/s", actual_tps);

        let unaccounted = actual_time - total_gemv - total_norm;
        println!(
            "\nUnaccounted time: {:.2} µs ({:.1}%)",
            unaccounted,
            unaccounted / actual_time * 100.0
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
