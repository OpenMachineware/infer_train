// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Compare our GEMV kernel with llama.cpp's kernel directly

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

        println!("\n=== GEMV Kernel Performance Comparison ===");

        let hidden_size = ctx.config.hidden_size;
        let layer = &ctx.model.layers[0];

        let wq = layer.wq.as_ref().expect("No wq");
        let q_size = wq.shape[1]; // 2048

        // Create input buffer
        let input_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let q_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (q_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let iterations = 100;

        // Measure GEMV timing
        let mut times_us: Vec<f64> = Vec::new();
        for _ in 0..iterations {
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
            times_us.push(start.elapsed().as_micros() as f64);
        }

        times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_us = times_us[iterations / 2];
        let avg_us: f64 = times_us.iter().sum::<f64>() / iterations as f64;

        println!("\nQ GEMV (M={}, K={}):", q_size, hidden_size);
        println!("  Median: {:.1} µs", median_us);
        println!("  Avg: {:.1} µs", avg_us);

        // Calculate bandwidth
        let weight_bytes = wq.data.as_bytes().len();
        let input_bytes = hidden_size * 4;
        let output_bytes = q_size * 4;
        let total_bytes = weight_bytes + input_bytes + output_bytes;

        println!("\nMemory access:");
        println!("  Weight: {:.2} MB", weight_bytes as f64 / 1e6);
        println!("  Input+Output: {} bytes", input_bytes + output_bytes);
        println!("  Total: {:.2} MB", total_bytes as f64 / 1e6);

        let bandwidth_gbs = (total_bytes as f64) / (median_us * 1e-6) / 1e9;
        println!("\nBandwidth: {:.1} GB/s", bandwidth_gbs);
        println!(
            "Utilization (100 GB/s peak): {:.1}%",
            bandwidth_gbs / 100.0 * 100.0
        );

        // Compare with decode
        println!("\n=== Full Decode ===");
        ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);

        let mut decode_times: Vec<f64> = Vec::new();
        for _ in 0..30 {
            let start = Instant::now();
            ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);
            decode_times.push(start.elapsed().as_micros() as f64);
        }

        decode_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let decode_median_us = decode_times[15];

        println!("Decode median: {:.1} µs", decode_median_us);
        println!("Tokens/second: {:.1}", 1_000_000.0 / decode_median_us);

        // Estimate total weight bytes
        let n_layers = ctx.model.layers.len();
        let per_layer_weight_bytes = {
            let wo = layer.wo.as_ref().expect("No wo");
            let gate = layer.w1.as_ref().expect("No gate");
            let up = layer.w3.as_ref().expect("No up");
            let down = layer.w2.as_ref().expect("No down");
            let wk = layer.wk.as_ref().expect("No wk");
            let wv = layer.wv.as_ref().expect("No wv");
            wq.data.as_bytes().len()
                + wk.data.as_bytes().len()
                + wv.data.as_bytes().len()
                + wo.data.as_bytes().len()
                + gate.data.as_bytes().len()
                + up.data.as_bytes().len()
                + down.data.as_bytes().len()
        };

        let total_weight_bytes = per_layer_weight_bytes * n_layers;
        println!("\nTotal weight: {:.2} MB", total_weight_bytes as f64 / 1e6);

        let decode_bandwidth_gbs =
            (total_weight_bytes as f64) / (decode_median_us * 1e-6) / 1e9;
        println!("Decode bandwidth: {:.1} GB/s", decode_bandwidth_gbs);
        println!("Utilization: {:.1}%", decode_bandwidth_gbs / 100.0 * 100.0);

        println!("\n=== Comparison with llama.cpp ===");
        println!("llama.cpp: 184 t/s (5.43 ms)");
        println!(
            "Ours: {:.1} t/s ({:.1} ms)",
            1_000_000.0 / decode_median_us,
            decode_median_us / 1000.0
        );
        println!("Gap: {:.2}x", 184.0 / (1_000_000.0 / decode_median_us));

        // Memory bandwidth comparison
        println!("\n=== Bandwidth Analysis ===");
        println!(
            "Single GEMV: {:.1} GB/s ({:.1}%)",
            bandwidth_gbs, bandwidth_gbs
        );
        println!(
            "Decode: {:.1} GB/s ({:.1}%)",
            decode_bandwidth_gbs, decode_bandwidth_gbs
        );
        println!(
            "Ratio: {:.2}x (overlap factor)",
            decode_bandwidth_gbs / bandwidth_gbs
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
