// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Instrumented decode profiling - measure actual kernel execution times

use std::time::Instant;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::dag::Operation;

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use metal::MTLResourceOptions;
        use std::rc::Rc;

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

        println!("\n=== Instrumented Decode Profiling ===");

        // Manual layer-by-layer execution with timing
        let n_layers = ctx.model.layers.len();
        let hidden_size = ctx.config.hidden_size;

        // Create input buffer (embedding output)
        let input_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        // Pre-allocate output buffers
        let norm_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let n_heads = ctx.config.n_heads;
        let head_dim = ctx.config.head_dim;
        let n_kv_heads = ctx.config.n_kv_heads;
        let q_size = n_heads * head_dim;
        let kv_size = n_kv_heads * head_dim;

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

        let rope_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (q_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let attn_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (q_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let wo_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let intermediate_size = ctx.config.intermediate_size;
        let gate_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (intermediate_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let up_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (intermediate_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let ffn_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        // Timing statistics
        let mut total_decode_us: u64 = 0;
        let mut layer_times: Vec<u64> = Vec::new();
        let mut gemv_times: Vec<u64> = Vec::new();
        let mut norm_times: Vec<u64> = Vec::new();
        let mut rope_times: Vec<u64> = Vec::new();
        let mut flash_attn_times: Vec<u64> = Vec::new();
        let mut ffn_times: Vec<u64> = Vec::new();

        // Test with actual decode
        println!("\nRunning instrumented decode...");

        for iter in 0..5 {
            let iter_start = Instant::now();

            // Full decode
            ctx.model.forward_gpu_native_batched(&[1], &mut kv_cache, &gpu_ctx);

            let iter_time = iter_start.elapsed().as_micros() as u64;
            if iter > 0 {
                // Skip warmup
                total_decode_us += iter_time;
            }
        }

        let avg_decode_us = total_decode_us / 4;
        println!(
            "\nAverage decode time: {} µs ({:.2} ms)",
            avg_decode_us,
            avg_decode_us as f64 / 1000.0
        );
        println!("Tokens/second: {:.1}", 1_000_000.0 / avg_decode_us as f64);

        // Now profile individual operations
        println!("\n=== Individual Operation Timing ===");

        let layer = &ctx.model.layers[0];
        let iterations = 20;

        // Time RMS Norm
        let norm = layer.attention_norm.as_ref().expect("No norm");
        let norm_buffer = gpu_ctx.get_or_create_buffer(norm);
        let mut times: Vec<u64> = Vec::new();
        for _ in 0..iterations {
            let start = Instant::now();
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
            times.push(start.elapsed().as_micros() as u64);
        }
        let avg_norm = times.iter().sum::<u64>() / times.len() as u64;
        println!("RMS Norm: {} µs", avg_norm);
        norm_times.push(avg_norm);

        // Time GEMV (Q projection - largest)
        let wq = layer.wq.as_ref().expect("No wq");
        let mut times: Vec<u64> = Vec::new();
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
            times.push(start.elapsed().as_micros() as u64);
        }
        let avg_gemv_q = times.iter().sum::<u64>() / times.len() as u64;
        println!("GEMV (Q projection): {} µs", avg_gemv_q);
        gemv_times.push(avg_gemv_q);

        // Time RoPE
        let mut times: Vec<u64> = Vec::new();
        for _ in 0..iterations {
            let start = Instant::now();
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();
            gpu_ctx.rope_neox_encode(
                &encoder,
                &q_out,
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
            times.push(start.elapsed().as_micros() as u64);
        }
        let avg_rope = times.iter().sum::<u64>() / times.len() as u64;
        println!("RoPE: {} µs", avg_rope);
        rope_times.push(avg_rope);

        // Time Flash Attention
        let seq_len = 101;
        let kv_stride = seq_len * head_dim * 2;
        let k_cache = kv_cache.k_buffer(0);
        let v_cache = kv_cache.v_buffer(0);

        let mut times: Vec<u64> = Vec::new();
        for _ in 0..iterations {
            let start = Instant::now();
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();
            gpu_ctx.flash_attn_mma_batched_encode(
                &encoder, n_heads, n_kv_heads, seq_len, kv_stride, &rope_out,
                k_cache, v_cache, &attn_out,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            times.push(start.elapsed().as_micros() as u64);
        }
        let avg_flash = times.iter().sum::<u64>() / times.len() as u64;
        println!("Flash Attention (seq_len={}): {} µs", seq_len, avg_flash);
        flash_attn_times.push(avg_flash);

        // Time FFN (Gate + Up + Down combined)
        let gate = layer.w1.as_ref().expect("No gate");
        let up = layer.w3.as_ref().expect("No up");
        let down = layer.w2.as_ref().expect("No down");
        let gate_size = gate.shape[1];
        let up_size = up.shape[1];

        let mut times: Vec<u64> = Vec::new();
        for _ in 0..iterations {
            let start = Instant::now();
            let cb = queue.new_command_buffer_with_unretained_references();
            // Gate
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                gate,
                gate_size,
                &gpu_ctx,
                &gate_out,
            );
            encoder.end_encoding();
            // Up
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &input_buffer,
                hidden_size,
                up,
                up_size,
                &gpu_ctx,
                &up_out,
            );
            encoder.end_encoding();
            // Down
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &gate_out,
                intermediate_size,
                down,
                hidden_size,
                &gpu_ctx,
                &ffn_out,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            times.push(start.elapsed().as_micros() as u64);
        }
        let avg_ffn = times.iter().sum::<u64>() / times.len() as u64;
        println!("FFN (Gate+Up+Down): {} µs", avg_ffn);
        ffn_times.push(avg_ffn);

        // Calculate theoretical decode time
        println!("\n=== Theoretical Analysis ===");
        println!("Per-layer operations:");
        println!("  1 Norm: {} µs", avg_norm);
        println!("  1 Q GEMV: {} µs", avg_gemv_q);
        println!("  2 KV GEMV: ~{} µs", avg_gemv_q * 2); // Estimate
        println!("  2 RoPE: {} µs", avg_rope * 2);
        println!("  1 Flash Attention: {} µs", avg_flash);
        println!("  1 Output GEMV: ~{} µs", avg_gemv_q);
        println!("  FFN: {} µs", avg_ffn);

        let per_layer_sequential =
            avg_norm + avg_gemv_q * 4 + avg_rope * 2 + avg_flash + avg_ffn;
        println!(
            "\nPer layer (sequential estimate): {} µs",
            per_layer_sequential
        );
        println!(
            "Total (28 layers, sequential): {} µs ({:.2} ms)",
            per_layer_sequential * 28,
            per_layer_sequential as f64 * 28.0 / 1000.0
        );

        // Actual decode time
        println!("\nActual decode: {} µs", avg_decode_us);
        println!(
            "Overlap factor: {:.2}x",
            per_layer_sequential as f64 * 28.0 / avg_decode_us as f64
        );

        // Time breakdown
        println!("\n=== Time Breakdown ===");
        let total_gemv_time = avg_gemv_q * 7 * 28; // 7 GEMV per layer
        let total_norm_time = avg_norm * 2 * 28; // 2 norm per layer
        let total_rope_time = avg_rope * 2 * 28;
        let total_flash_time = avg_flash * 28;
        let total_ffn_time = avg_ffn * 28;

        println!("GEMV (sequential): {} µs", total_gemv_time);
        println!("Norm (sequential): {} µs", total_norm_time);
        println!("RoPE (sequential): {} µs", total_rope_time);
        println!("Flash Attention (sequential): {} µs", total_flash_time);
        println!("FFN (sequential): {} µs", total_ffn_time);

        let total_sequential = total_gemv_time
            + total_norm_time
            + total_rope_time
            + total_flash_time
            + total_ffn_time;
        println!(
            "Total sequential: {} µs ({:.2} ms)",
            total_sequential,
            total_sequential as f64 / 1000.0
        );

        println!("\nActual decode: {} µs", avg_decode_us);
        println!(
            "Achieved overlap: {:.2}x",
            total_sequential as f64 / avg_decode_us as f64
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
