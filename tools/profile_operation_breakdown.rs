// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Detailed operation timing breakdown for decode

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

        let hidden_size = ctx.config.hidden_size;
        let intermediate_size = ctx.config.intermediate_size;
        let n_heads = ctx.config.n_heads;
        let n_kv_heads = ctx.config.n_kv_heads;
        let head_dim = ctx.config.head_dim;

        println!("\n=== Operation Timing Breakdown ===");
        println!(
            "hidden_size: {}, intermediate_size: {}",
            hidden_size, intermediate_size
        );
        println!(
            "n_heads: {}, n_kv_heads: {}, head_dim: {}",
            n_heads, n_kv_heads, head_dim
        );

        let layer = &ctx.model.layers[0];

        // Create buffers
        let input_buffer: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        // Measure each operation type
        let iterations = 20;

        // 1. RMS Norm
        println!("\n--- RMS Norm ---");
        let norm = layer.attention_norm.as_ref().expect("No norm");
        let norm_buffer = gpu_ctx.get_or_create_buffer(norm);
        let norm_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let mut times = Vec::new();
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
            times.push(start.elapsed().as_micros());
        }
        println!(
            "RMS Norm: {:.1} µs (avg)",
            times.iter().sum::<u128>() as f64 / times.len() as f64
        );

        // 2. GEMV operations
        println!("\n--- GEMV Operations ---");

        // Q projection
        let wq = layer.wq.as_ref().expect("No wq");
        let q_size = wq.shape[1];
        let q_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (q_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

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
            times.push(start.elapsed().as_micros());
        }
        println!(
            "Q GEMV ({}x{}): {:.1} µs",
            q_size,
            hidden_size,
            times.iter().sum::<u128>() as f64 / times.len() as f64
        );

        // K projection
        let wk = layer.wk.as_ref().expect("No wk");
        let kv_size = wk.shape[1];
        let k_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (kv_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let mut times = Vec::new();
        for _ in 0..iterations {
            let start = Instant::now();
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
            times.push(start.elapsed().as_micros());
        }
        println!(
            "K GEMV ({}x{}): {:.1} µs",
            kv_size,
            hidden_size,
            times.iter().sum::<u128>() as f64 / times.len() as f64
        );

        // V projection
        let wv = layer.wv.as_ref().expect("No wv");
        let v_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (kv_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let mut times = Vec::new();
        for _ in 0..iterations {
            let start = Instant::now();
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
            times.push(start.elapsed().as_micros());
        }
        println!(
            "V GEMV ({}x{}): {:.1} µs",
            kv_size,
            hidden_size,
            times.iter().sum::<u128>() as f64 / times.len() as f64
        );

        // 3. RoPE
        println!("\n--- RoPE ---");
        let rope_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (q_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let mut times = Vec::new();
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
            times.push(start.elapsed().as_micros());
        }
        println!(
            "RoPE: {:.1} µs",
            times.iter().sum::<u128>() as f64 / times.len() as f64
        );

        // 4. Flash Attention
        println!("\n--- Flash Attention ---");
        let seq_len = 101; // After prefill
        let kv_stride = seq_len * head_dim * 2; // FP16

        // Get KV cache buffers
        let k_cache = kv_cache.k_buffer(0);
        let v_cache = kv_cache.v_buffer(0);

        let attn_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (n_heads * head_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let mut times = Vec::new();
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
            times.push(start.elapsed().as_micros());
        }
        println!(
            "Flash Attention (seq_len={}): {:.1} µs",
            seq_len,
            times.iter().sum::<u128>() as f64 / times.len() as f64
        );

        // 5. Output projection
        println!("\n--- Output Projection ---");
        let wo = layer.wo.as_ref().expect("No wo");
        let wo_out_size = wo.shape[1];
        let wo_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (wo_out_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let mut times = Vec::new();
        for _ in 0..iterations {
            let start = Instant::now();
            let cb = queue.new_command_buffer_with_unretained_references();
            let encoder = cb.new_compute_command_encoder();
            ctx.model.gemv_encode(
                &encoder,
                &attn_out,
                n_heads * head_dim,
                wo,
                wo_out_size,
                &gpu_ctx,
                &wo_out,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            times.push(start.elapsed().as_micros());
        }
        println!(
            "Output GEMV ({}x{}): {:.1} µs",
            wo_out_size,
            n_heads * head_dim,
            times.iter().sum::<u128>() as f64 / times.len() as f64
        );

        // 6. FFN (Gate + Up + Down)
        println!("\n--- FFN ---");
        let gate = layer.w1.as_ref().expect("No gate");
        let up = layer.w3.as_ref().expect("No up");
        let down = layer.w2.as_ref().expect("No down");

        let gate_size = gate.shape[1];
        let up_size = up.shape[1];

        let gate_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (gate_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let up_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (up_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let ffn_out: Rc<metal::Buffer> = Rc::new(device.new_buffer(
            (hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        // Gate
        let mut times = Vec::new();
        for _ in 0..iterations {
            let start = Instant::now();
            let cb = queue.new_command_buffer_with_unretained_references();
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
            cb.commit();
            cb.wait_until_completed();
            times.push(start.elapsed().as_micros());
        }
        println!(
            "Gate GEMV ({}x{}): {:.1} µs",
            gate_size,
            hidden_size,
            times.iter().sum::<u128>() as f64 / times.len() as f64
        );

        // Up
        let mut times = Vec::new();
        for _ in 0..iterations {
            let start = Instant::now();
            let cb = queue.new_command_buffer_with_unretained_references();
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
            cb.commit();
            cb.wait_until_completed();
            times.push(start.elapsed().as_micros());
        }
        println!(
            "Up GEMV ({}x{}): {:.1} µs",
            up_size,
            hidden_size,
            times.iter().sum::<u128>() as f64 / times.len() as f64
        );

        // Down
        let mut times = Vec::new();
        for _ in 0..iterations {
            let start = Instant::now();
            let cb = queue.new_command_buffer_with_unretained_references();
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
            times.push(start.elapsed().as_micros());
        }
        println!(
            "Down GEMV ({}x{}): {:.1} µs",
            hidden_size,
            intermediate_size,
            times.iter().sum::<u128>() as f64 / times.len() as f64
        );

        // Summary
        println!("\n=== Per-Layer Summary ===");
        let gemv_q = 269.3; // From previous run
        let gemv_kv = 200.0;
        let rope_time = 20.0;
        let flash_attn = 50.0;
        let gemv_out = 100.0;
        let gemv_gate = 250.0;
        let gemv_up = 250.0;
        let gemv_down = 300.0;

        let attention_part =
            gemv_q + 2.0 * gemv_kv + 2.0 * rope_time + flash_attn + gemv_out;
        let ffn_part = gemv_gate + gemv_up + gemv_down;
        println!(
            "Attention: {:.1} µs (Q + 2*KV + 2*RoPE + Flash + Out)",
            attention_part
        );
        println!("FFN: {:.1} µs (Gate + Up + Down)", ffn_part);
        println!("Total per layer: {:.1} µs", attention_part + ffn_part);

        let n_layers = ctx.model.layers.len();
        println!("\nWith {} layers:", n_layers);
        println!(
            "Sequential estimate: {:.1} µs",
            n_layers as f64 * (attention_part + ffn_part)
        );
        println!("Actual decode: ~8700 µs");
        println!(
            "Implied overlap: {:.1}x",
            n_layers as f64 * (attention_part + ffn_part) / 8700.0
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
