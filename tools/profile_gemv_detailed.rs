// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Detailed GEMV kernel analysis: compare our implementation with theoretical peak

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

        println!("\n=== GEMV Kernel Detailed Analysis ===");

        let hidden_size = ctx.config.hidden_size;
        let layer = &ctx.model.layers[0];

        // Create input buffer
        let input_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let iterations = 100;

        // Test Q projection (largest GEMV)
        let wq = layer.wq.as_ref().expect("No wq");
        let q_size = wq.shape[1]; // 2048

        let q_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (q_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        // Measure GEMV timing with high precision
        let mut times = Vec::new();
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
            times.push(start.elapsed().as_nanos());
        }

        // Calculate statistics
        times.sort();
        let min_ns = times[0];
        let median_ns = times[iterations / 2];
        let avg_ns: f64 = times.iter().sum::<u128>() as f64 / iterations as f64;

        println!("\nQ GEMV (M={}, K={}):", q_size, hidden_size);
        println!("  Min: {} ns", min_ns);
        println!("  Median: {} ns", median_ns);
        println!("  Avg: {} ns", avg_ns);
        println!("  Avg: {:.1} µs", avg_ns / 1000.0);

        // Calculate bandwidth
        let weight_bytes = wq.data.as_bytes().len();
        let input_bytes = hidden_size * 4;
        let output_bytes = q_size * 4;
        let total_bytes = weight_bytes + input_bytes + output_bytes;

        println!("\nMemory access:");
        println!(
            "  Weight: {} bytes ({:.2} MB)",
            weight_bytes,
            weight_bytes as f64 / 1e6
        );
        println!("  Input: {} bytes", input_bytes);
        println!("  Output: {} bytes", output_bytes);
        println!(
            "  Total: {} bytes ({:.2} MB)",
            total_bytes,
            total_bytes as f64 / 1e6
        );

        let bandwidth_gb_s = (total_bytes as f64 / avg_ns) * 1000.0; // bytes/ns -> GB/s
        println!("\nBandwidth: {:.1} GB/s", bandwidth_gb_s);
        println!(
            "Utilization (100 GB/s peak): {:.1}%",
            bandwidth_gb_s / 100.0 * 100.0
        );

        // Test with multiple GEMV in one command buffer
        println!("\n--- Multiple GEMV in one CB ---");

        // 3 GEMV: Q, K, V
        let wk = layer.wk.as_ref().expect("No wk");
        let wv = layer.wv.as_ref().expect("No wv");
        let kv_size = wk.shape[1];

        let k_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (kv_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let v_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (kv_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        // Measure 3 GEMV batched
        let mut times = Vec::new();
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
            times.push(start.elapsed().as_nanos());
        }

        times.sort();
        let qkv_median_ns = times[iterations / 2];

        println!("\n3 GEMV (Q+K+V) batched:");
        println!(
            "  Median: {} ns ({:.1} µs)",
            qkv_median_ns,
            qkv_median_ns as f64 / 1000.0
        );

        let qkv_total_bytes = wq.data.as_bytes().len()
            + wk.data.as_bytes().len()
            + wv.data.as_bytes().len()
            + 3 * (input_bytes + output_bytes);
        let qkv_bandwidth =
            (qkv_total_bytes as f64 / qkv_median_ns as f64) * 1000.0;
        println!("  Total bytes: {:.2} MB", qkv_total_bytes as f64 / 1e6);
        println!("  Bandwidth: {:.1} GB/s", qkv_bandwidth);
        println!("  Utilization: {:.1}%", qkv_bandwidth / 100.0 * 100.0);

        // Compare with 3 separate GEMV
        let mut times = Vec::new();
        for _ in 0..iterations {
            let start = Instant::now();
            // Q
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
            // K
            let cb = queue.new_command_buffer_with_unretained_references();
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
            cb.commit();
            cb.wait_until_completed();
            // V
            let cb = queue.new_command_buffer_with_unretained_references();
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
            times.push(start.elapsed().as_nanos());
        }

        times.sort();
        let qkv_separate_ns = times[iterations / 2];

        println!("\n3 GEMV (Q+K+V) separate:");
        println!(
            "  Median: {} ns ({:.1} µs)",
            qkv_separate_ns,
            qkv_separate_ns as f64 / 1000.0
        );
        println!(
            "  Batched speedup: {:.2}x",
            qkv_separate_ns as f64 / qkv_median_ns as f64
        );

        // Now compare with full decode
        println!("\n--- Full Decode Analysis ---");

        // Warmup
        ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);

        let mut times = Vec::new();
        for _ in 0..20 {
            let start = Instant::now();
            ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);
            times.push(start.elapsed().as_nanos());
        }

        times.sort();
        let decode_median_ns = times[10];

        println!(
            "\nFull decode median: {} ns ({:.1} µs)",
            decode_median_ns,
            decode_median_ns as f64 / 1000.0
        );

        // Estimate total memory access for full decode
        let n_layers = ctx.model.layers.len();
        let per_layer_weight_bytes = {
            let wo = layer.wo.as_ref().expect("No wo");
            let gate = layer.w1.as_ref().expect("No gate");
            let up = layer.w3.as_ref().expect("No up");
            let down = layer.w2.as_ref().expect("No down");
            wq.data.as_bytes().len()
                + wk.data.as_bytes().len()
                + wv.data.as_bytes().len()
                + wo.data.as_bytes().len()
                + gate.data.as_bytes().len()
                + up.data.as_bytes().len()
                + down.data.as_bytes().len()
        };
        let total_weight_bytes = per_layer_weight_bytes * n_layers;

        println!(
            "\nTotal weight bytes: {:.2} MB",
            total_weight_bytes as f64 / 1e6
        );

        let decode_bandwidth =
            (total_weight_bytes as f64 / decode_median_ns as f64) * 1000.0;
        println!("Decode bandwidth: {:.1} GB/s", decode_bandwidth);
        println!("Utilization: {:.1}%", decode_bandwidth / 100.0 * 100.0);

        println!("\n=== Comparison with llama.cpp ===");
        println!("llama.cpp decode: 184 t/s = 5.43 ms per token");
        println!(
            "Our decode: {:.1} µs per token",
            decode_median_ns as f64 / 1000.0
        );
        println!("Gap: {:.2}x", (decode_median_ns as f64) / 5430.0 / 1000.0);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
