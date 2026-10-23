// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Profile decode execution to understand operation batching

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
        println!("=== Prefilling ===");
        let prefill_tokens: Vec<u32> = (0..100).collect();
        ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv_cache,
            &gpu_ctx,
        );

        println!("\n=== Decode Breakdown Analysis ===");
        println!("Model: {} layers", ctx.model.layers.len());

        // Count operations per layer
        let layer = &ctx.model.layers[0];
        println!("\nPer-layer operations:");
        println!("  - RMS Norm (attention_norm)");
        println!("  - Q projection (GEMV)");
        println!("  - K projection (GEMV)");
        println!("  - V projection (GEMV)");
        println!("  - RoPE for Q");
        println!("  - RoPE for K");
        println!("  - KV Cache update for K");
        println!("  - KV Cache update for V");
        println!("  - Flash Attention");
        println!("  - Output projection (GEMV)");
        println!("  - RMS Norm (ffn_norm)");
        println!("  - Gate projection (GEMV)");
        println!("  - Up projection (GEMV)");
        println!("  - SiLU activation");
        println!("  - Element-wise multiply");
        println!("  - Down projection (GEMV)");
        println!("  ~17 operations per layer");

        let n_layers = ctx.model.layers.len();
        println!(
            "Total: ~{} operations for {} layers",
            n_layers * 17,
            n_layers
        );

        // Measure timing breakdown
        let iterations = 10;
        let mut total_decode_time = 0u128;

        // Warm up
        ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);

        // Measure total decode
        println!("\n=== Total Decode Timing ===");
        for _ in 0..iterations {
            let start = Instant::now();
            ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);
            total_decode_time += start.elapsed().as_micros();
        }
        let avg_decode = total_decode_time as f64 / iterations as f64;
        println!("Average decode time: {:.1} µs", avg_decode);

        // Measure individual kernels
        println!("\n=== Individual Kernel Timing ===");

        let hidden_size = ctx.config.hidden_size;
        let intermediate_size = ctx.config.intermediate_size;

        // Create test buffers
        let input_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let wq = layer.wq.as_ref().expect("No wq");
        let q_size = wq.shape[1];

        // Measure single GEMV
        let gemv_times: Vec<u128> = (0..20)
            .map(|_| {
                let q_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
                    (q_size * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                ));
                let start = Instant::now();
                let cb = queue.new_command_buffer_with_unretained_references();
                let encoder = cb.new_compute_command_encoder();
                ctx.model.gemv_encode(
                    &encoder,
                    &input_buffer,
                    hidden_size,
                    wq,
                    q_size,
                    &gpu_ctx,
                    &q_out,
                );
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                start.elapsed().as_micros()
            })
            .collect();
        let avg_gemv =
            gemv_times.iter().sum::<u128>() as f64 / gemv_times.len() as f64;
        println!("Single GEMV: {:.1} µs", avg_gemv);

        // Measure 3 GEMV batched (Q, K, V projections)
        let wk = layer.wk.as_ref().expect("No wk");
        let wv = layer.wv.as_ref().expect("No wv");
        let kv_size = wk.shape[1];

        let qkv_times: Vec<u128> = (0..20)
            .map(|_| {
                let q_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
                    (q_size * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                ));
                let k_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
                    (kv_size * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                ));
                let v_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
                    (kv_size * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                ));
                let start = Instant::now();
                let cb = queue.new_command_buffer_with_unretained_references();
                let encoder = cb.new_compute_command_encoder();
                ctx.model.gemv_encode(
                    &encoder,
                    &input_buffer,
                    hidden_size,
                    wq,
                    q_size,
                    &gpu_ctx,
                    &q_out,
                );
                encoder.end_encoding();
                let encoder = cb.new_compute_command_encoder();
                ctx.model.gemv_encode(
                    &encoder,
                    &input_buffer,
                    hidden_size,
                    wk,
                    kv_size,
                    &gpu_ctx,
                    &k_out,
                );
                encoder.end_encoding();
                let encoder = cb.new_compute_command_encoder();
                ctx.model.gemv_encode(
                    &encoder,
                    &input_buffer,
                    hidden_size,
                    wv,
                    kv_size,
                    &gpu_ctx,
                    &v_out,
                );
                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                start.elapsed().as_micros()
            })
            .collect();
        let avg_qkv =
            qkv_times.iter().sum::<u128>() as f64 / qkv_times.len() as f64;
        println!("3 GEMV (Q+K+V) batched: {:.1} µs", avg_qkv);
        println!(
            "  (vs 3x single: {:.1} µs, speedup: {:.2}x)",
            3.0 * avg_gemv,
            3.0 * avg_gemv / avg_qkv
        );

        // Estimate total GEMV time
        // Per layer: Q, K, V, output, gate, up, down = 7 GEMV
        // But Q, K, V can be batched, gate+up can be batched
        let total_gemv_sequential = n_layers as f64 * 7.0 * avg_gemv;
        println!(
            "\nTotal GEMV estimate (sequential): {:.1} µs",
            total_gemv_sequential
        );

        // What we'd expect with perfect batching
        // Per layer: batch(Q,K,V) + output + batch(gate,up) + down
        // = avg_qkv + avg_gemv + 2*avg_gemv + avg_gemv = avg_qkv + 4*avg_gemv
        let ideal_per_layer = avg_qkv + 4.0 * avg_gemv;
        let ideal_total = n_layers as f64 * ideal_per_layer;
        println!("Total GEMV estimate (ideal batching): {:.1} µs", ideal_total);
        println!(
            "  Per layer: QKV({:.1}) + out({:.1}) + gate_up(2*{:.1}) + down({:.1}) = {:.1} µs",
            avg_qkv, avg_gemv, avg_gemv, avg_gemv, ideal_per_layer
        );

        // Compare with actual
        println!("\n=== Analysis ===");
        println!("Actual decode: {:.1} µs", avg_decode);
        println!(
            "Ideal GEMV only: {:.1} µs ({:.1}%)",
            ideal_total,
            ideal_total / avg_decode * 100.0
        );
        println!(
            "Non-GEMV time: {:.1} µs ({:.1}%)",
            avg_decode - ideal_total,
            (avg_decode - ideal_total) / avg_decode * 100.0
        );

        // The gap
        let gap = avg_decode - ideal_total;
        println!("\nGap analysis:");
        println!(
            "  - Flash Attention: ~{:.0} µs (based on previous profiling)",
            200.0
        );
        println!(
            "  - RoPE operations: ~{:.0} µs",
            n_layers as f64 * 2.0 * 20.0
        );
        println!("  - KV updates: ~{:.0} µs", n_layers as f64 * 2.0 * 15.0);
        println!(
            "  - Remaining unaccounted: {:.0} µs",
            gap - 200.0 - n_layers as f64 * 70.0
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
