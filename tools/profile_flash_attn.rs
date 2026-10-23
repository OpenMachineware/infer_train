// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Profile Flash Attention performance during decode

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

        let n_layers = ctx.model.layers.len();
        let n_heads = ctx.config.n_heads;
        let n_kv_heads = ctx.config.n_kv_heads;
        let head_dim = ctx.config.head_dim;
        let seq_len = 101; // After prefill

        println!("\n=== Flash Attention Profile ===");
        println!(
            "Layers: {}, n_heads: {}, n_kv_heads: {}, head_dim: {}",
            n_layers, n_heads, n_kv_heads, head_dim
        );
        println!("seq_len: {}", seq_len);

        // Create Q buffer
        let q_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (n_heads * head_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let iterations = 10;

        // Test single Flash Attention
        println!("\n--- Single Flash Attention (per layer) ---");
        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();

            let k_buf = kv_cache.k_buffer(0);
            let v_buf = kv_cache.v_buffer(0);
            let output_buf: Rc<metal::Buffer> = Rc::new(device.new_buffer(
                (n_heads * head_dim * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            gpu_ctx.flash_attn_f16_encode(
                encoder,
                1, // n_seqs
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                kv_cache.max_seq(),
                &q_buffer,
                k_buf,
                v_buf,
                &output_buf,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let fa_single_time =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("Flash Attention time: {:.2} µs", fa_single_time);

        // Test all layers Flash Attention (sequential CBs)
        println!("\n--- All Layers Flash Attention (sequential CBs) ---");
        let start = Instant::now();
        for _ in 0..iterations {
            for layer_idx in 0..n_layers {
                let cb = queue.new_command_buffer_with_unretained_references();
                let encoder = cb.new_compute_command_encoder();

                let k_buf = kv_cache.k_buffer(layer_idx);
                let v_buf = kv_cache.v_buffer(layer_idx);
                let output_buf: Rc<metal::Buffer> = Rc::new(device.new_buffer(
                    (n_heads * head_dim * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                ));

                gpu_ctx.flash_attn_f16_encode(
                    encoder,
                    1,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    seq_len,
                    kv_cache.max_seq(),
                    &q_buffer,
                    k_buf,
                    v_buf,
                    &output_buf,
                );

                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            }
        }
        let fa_all_sequential =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("Total time (sequential CBs): {:.2} µs", fa_all_sequential);
        println!("Per layer: {:.2} µs", fa_all_sequential / n_layers as f64);

        // Test all layers Flash Attention (single CB)
        println!("\n--- All Layers Flash Attention (single CB) ---");
        let start = Instant::now();
        for _ in 0..iterations {
            let cb = queue.new_command_buffer_with_unretained_references();

            for layer_idx in 0..n_layers {
                let encoder = cb.new_compute_command_encoder();

                let k_buf = kv_cache.k_buffer(layer_idx);
                let v_buf = kv_cache.v_buffer(layer_idx);
                let output_buf: Rc<metal::Buffer> = Rc::new(device.new_buffer(
                    (n_heads * head_dim * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                ));

                gpu_ctx.flash_attn_f16_encode(
                    encoder,
                    1,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    seq_len,
                    kv_cache.max_seq(),
                    &q_buffer,
                    k_buf,
                    v_buf,
                    &output_buf,
                );

                encoder.end_encoding();
            }

            cb.commit();
            cb.wait_until_completed();
        }
        let fa_all_batched =
            start.elapsed().as_micros() as f64 / iterations as f64;
        println!("Total time (single CB): {:.2} µs", fa_all_batched);
        println!("Per layer: {:.2} µs", fa_all_batched / n_layers as f64);

        let overlap_speedup = fa_all_sequential / fa_all_batched;
        println!("Batched speedup: {:.2}x", overlap_speedup);

        // Compare with decode time
        println!("\n=== Comparison with Decode ===");
        let start = Instant::now();
        for _ in 0..10 {
            ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);
        }
        let decode_time = start.elapsed().as_micros() as f64 / 10.0;

        println!("Full decode time: {:.2} µs", decode_time);
        println!("Flash Attention total: {:.2} µs", fa_all_batched);
        println!(
            "Flash Attention %: {:.1}%",
            fa_all_batched / decode_time * 100.0
        );

        // Calculate bandwidth for Flash Attention
        let kv_bytes = seq_len * n_kv_heads * head_dim * 4 * 2; // K + V
        let q_bytes = n_heads * head_dim * 4;
        let output_bytes = n_heads * head_dim * 4;
        let total_fa_bytes = (kv_bytes + q_bytes + output_bytes) as f64;

        let fa_bandwidth = total_fa_bytes / (fa_single_time * 1e-6) / 1e9;
        let fa_util = fa_bandwidth / 100.0 * 100.0;

        println!("\n=== Flash Attention Bandwidth ===");
        println!("KV cache: {} KB", kv_bytes / 1024);
        println!("Q + Output: {} KB", (q_bytes + output_bytes) / 1024);
        println!("Total: {:.1} KB", total_fa_bytes / 1024.0);
        println!("Bandwidth: {:.1} GB/s", fa_bandwidth);
        println!("Utilization: {:.1}%", fa_util);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
