// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Profile kernel overlap efficiency
//!
//! Measures whether multiple kernels in one command buffer can overlap.
//! If they can, bandwidth should scale with kernel count.

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

        println!("\n=== Kernel Overlap Efficiency ===");
        println!("Model: Qwen3-0.6B Q4_K\n");

        let layer = &ctx.model.layers[0];
        let hidden_size = ctx.config.hidden_size;
        let wq = layer.wq.as_ref().expect("No wq");
        let q_out_size = wq.shape[1];

        // Create test buffers
        let input_buffer: Rc<metal::Buffer> =
            Rc::new(device.new_buffer_with_data(
                vec![1.0f32; hidden_size].as_ptr() as *const _,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));
        let out_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (q_out_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let iterations = 10;

        // Test 1: Single kernel per command buffer (sequential)
        println!("--- Test 1: Sequential Command Buffers ---");
        let start = Instant::now();
        for _ in 0..iterations {
            for _ in 0..6 {
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
        }
        let sequential_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("6 kernels, sequential CB: {:.2} µs", sequential_time);
        println!("Per kernel: {:.2} µs", sequential_time / 6.0);

        // Test 2: All kernels in one command buffer (potential overlap)
        println!("\n--- Test 2: Single Command Buffer (all kernels) ---");
        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            for _ in 0..6 {
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
            }
            cb.commit();
            cb.wait_until_completed();
        }
        let batched_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("6 kernels, single CB: {:.2} µs", batched_time);
        println!("Per kernel: {:.2} µs", batched_time / 6.0);

        let overlap_speedup = sequential_time / batched_time;
        println!("\nBatched speedup: {:.2}x", overlap_speedup);

        // Test 3: Different kernel counts
        println!("\n--- Test 3: Kernel Count Scaling ---");
        for count in [1, 2, 4, 8, 12, 16] {
            let start = Instant::now();
            for _ in 0..iterations {
                let cb = queue.new_command_buffer_with_unretained_references();
                for _ in 0..count {
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
                }
                cb.commit();
                cb.wait_until_completed();
            }
            let time = start.elapsed().as_micros() as f64 / iterations as f64;
            let per_kernel = time / count as f64;
            let theoretical = sequential_time / 6.0 * count as f64;
            let overlap_ratio = theoretical / time;
            println!(
                "{} kernels: {:7.1} µs total, {:6.1} µs/kernel, overlap: {:.2}x",
                count, time, per_kernel, overlap_ratio
            );
        }

        // Analysis
        println!("\n=== Analysis ===");
        if overlap_speedup > 1.5 {
            println!(
                "✅ Good kernel overlap - {:.1}x improvement from batching",
                overlap_speedup
            );
        } else {
            println!(
                "❌ Poor kernel overlap - only {:.1}x improvement",
                overlap_speedup
            );
            println!("   This suggests kernels cannot execute in parallel.");
            println!(
                "   Metal may serialize them due to resource constraints."
            );
        }

        // Calculate bandwidth
        let weight_bytes = q_out_size * hidden_size * (144 / 256);
        let single_kernel_bytes =
            weight_bytes + hidden_size * 4 + q_out_size * 4;
        let single_kernel_time = batched_time / 6.0;
        let bw_gbs =
            single_kernel_bytes as f64 / (single_kernel_time * 1e-6) / 1e9;
        let util = bw_gbs / 100.0 * 100.0;
        println!(
            "\nSingle kernel bandwidth: {:.1} GB/s ({:.1}%)",
            bw_gbs, util
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
