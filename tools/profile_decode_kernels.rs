// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile decode kernel performance for each layer

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = args[1].clone();

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::time::Instant;

        println!("=== Decode Kernel Performance Profile ===");
        println!("Model: {}\n", model_path);

        let result = std::thread::spawn(move || {
            let gpu_ctx = MetalContext::new().unwrap();
            let device = gpu_ctx.device();
            let ctx = InferenceContext::from_gguf_with_gpu_ctx(
                &model_path,
                gpu_ctx.clone(),
            )
            .unwrap();
            let config = ctx.model.config.clone();
            let n_layers = config.n_layers;
            let hidden_size = config.hidden_size;
            let n_heads = config.n_heads;
            let head_dim = config.head_dim;
            let intermediate_size = ctx.model.layers[0]
                .w1
                .as_ref()
                .map(|w| w.shape[1])
                .unwrap_or(0);

            println!(
                "Layers: {}, Hidden: {}, Heads: {}, Head dim: {}",
                n_layers, hidden_size, n_heads, head_dim
            );
            println!("Intermediate size: {}", intermediate_size);

            let mut kv = KVCacheGPU::new(device, &config, 2048);

            // Prefill
            let prefill_tokens: Vec<u32> = (1..=10).collect();
            let _ = ctx.model.forward_gpu_native_batched(
                &prefill_tokens,
                &mut kv,
                &gpu_ctx,
            );
            println!("\nPrefilled 10 tokens\n");

            // Test individual kernel performance
            use metal::Buffer;
            use std::cell::RefCell;
            use std::rc::Rc;

            let reuse_pool: Rc<RefCell<Vec<Rc<Buffer>>>> =
                Rc::new(RefCell::new(Vec::new()));
            let mut forward_pool = infer_train::buffer::ForwardBufferPool::new(
                device.clone(),
                reuse_pool.clone(),
            );

            // Get layer 0 weights
            let layer = &ctx.model.layers[0];
            let attention_norm = layer.attention_norm.as_ref().unwrap();
            let wq = layer.wq.as_ref().unwrap();
            let wk = layer.wk.as_ref().unwrap();
            let wv = layer.wv.as_ref().unwrap();
            let wo = layer.wo.as_ref().unwrap();
            let w1 = layer.w1.as_ref().unwrap();
            let w2 = layer.w2.as_ref().unwrap();
            let w3 = layer.w3.as_ref().unwrap();

            // Create input buffer
            let input_buffer = forward_pool.get(hidden_size * 4);

            println!("=== Kernel Performance (Layer 0) ===\n");

            // Test RMS Norm
            {
                let norm_buffer = forward_pool.get(hidden_size * 4);
                let norm_weight = gpu_ctx.get_or_create_buffer(attention_norm);

                let command_buffer = gpu_ctx.queue().new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                gpu_ctx.rms_norm_encode(
                    &encoder,
                    &input_buffer,
                    &norm_weight,
                    &norm_buffer,
                    1,
                    hidden_size,
                    1e-5,
                );
                encoder.end_encoding();

                let start = Instant::now();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                let elapsed = start.elapsed();

                println!("RMS Norm: {:?}", elapsed);
            }

            // Test Q projection
            {
                let q_out_size = wq.shape[1];
                let q_buffer = forward_pool.get(q_out_size * 4);

                let command_buffer = gpu_ctx.queue().new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
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

                let start = Instant::now();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                let elapsed = start.elapsed();

                println!(
                    "Q projection ({}x{}): {:?}",
                    hidden_size, q_out_size, elapsed
                );
            }

            // Test K projection
            {
                let k_out_size = wk.shape[1];
                let k_buffer = forward_pool.get(k_out_size * 4);

                let command_buffer = gpu_ctx.queue().new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                ctx.model.gemv_encode(
                    &encoder,
                    &input_buffer,
                    hidden_size,
                    wk,
                    k_out_size,
                    &gpu_ctx,
                    &k_buffer,
                );
                encoder.end_encoding();

                let start = Instant::now();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                let elapsed = start.elapsed();

                println!(
                    "K projection ({}x{}): {:?}",
                    hidden_size, k_out_size, elapsed
                );
            }

            // Test V projection
            {
                let v_out_size = wv.shape[1];
                let v_buffer = forward_pool.get(v_out_size * 4);

                let command_buffer = gpu_ctx.queue().new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                ctx.model.gemv_encode(
                    &encoder,
                    &input_buffer,
                    hidden_size,
                    wv,
                    v_out_size,
                    &gpu_ctx,
                    &v_buffer,
                );
                encoder.end_encoding();

                let start = Instant::now();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                let elapsed = start.elapsed();

                println!(
                    "V projection ({}x{}): {:?}",
                    hidden_size, v_out_size, elapsed
                );
            }

            // Test RoPE
            {
                let q_size = n_heads * head_dim;
                let q_buffer = forward_pool.get(q_size * 4);
                let q_rope = forward_pool.get(q_size * 4);

                let command_buffer = gpu_ctx.queue().new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                gpu_ctx.rope_neox_encode(
                    &encoder,
                    &q_buffer,
                    &[0],
                    1,
                    n_heads,
                    head_dim,
                    10000.0,
                    1.0,
                    &q_rope,
                );
                encoder.end_encoding();

                let start = Instant::now();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                let elapsed = start.elapsed();

                println!("RoPE: {:?}", elapsed);
            }

            // Test Flash Attention
            {
                let q_size = n_heads * head_dim;
                let q_buffer = forward_pool.get(q_size * 4);
                let k_cache = kv.k_buffer(0);
                let v_cache = kv.v_buffer(0);
                let attn_out = forward_pool.get(q_size * 4);

                let seq_k = kv.seq_len();
                let n_kv_heads = wk.shape[1] / head_dim;

                let command_buffer = gpu_ctx.queue().new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                gpu_ctx.flash_attn_f16_encode(
                    &encoder,
                    1,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    seq_k,
                    kv.max_seq(),
                    &q_buffer,
                    k_cache,
                    v_cache,
                    &attn_out,
                );
                encoder.end_encoding();

                let start = Instant::now();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                let elapsed = start.elapsed();

                println!("Flash Attention (seq_k={}): {:?}", seq_k, elapsed);
            }

            // Test Output projection
            {
                let wo_out_size = wo.shape[1];
                let wo_buffer = forward_pool.get(wo_out_size * 4);

                let command_buffer = gpu_ctx.queue().new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                ctx.model.gemv_encode(
                    &encoder,
                    &input_buffer,
                    hidden_size,
                    wo,
                    wo_out_size,
                    &gpu_ctx,
                    &wo_buffer,
                );
                encoder.end_encoding();

                let start = Instant::now();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                let elapsed = start.elapsed();

                println!(
                    "Output projection ({}x{}): {:?}",
                    hidden_size, wo_out_size, elapsed
                );
            }

            // Test FFN gate
            {
                let gate_buffer = forward_pool.get(intermediate_size * 4);

                let command_buffer = gpu_ctx.queue().new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                ctx.model.gemv_encode(
                    &encoder,
                    &input_buffer,
                    hidden_size,
                    w1,
                    intermediate_size,
                    &gpu_ctx,
                    &gate_buffer,
                );
                encoder.end_encoding();

                let start = Instant::now();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                let elapsed = start.elapsed();

                println!(
                    "FFN gate ({}x{}): {:?}",
                    hidden_size, intermediate_size, elapsed
                );
            }

            // Test FFN up
            {
                let up_buffer = forward_pool.get(intermediate_size * 4);

                let command_buffer = gpu_ctx.queue().new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                ctx.model.gemv_encode(
                    &encoder,
                    &input_buffer,
                    hidden_size,
                    w3,
                    intermediate_size,
                    &gpu_ctx,
                    &up_buffer,
                );
                encoder.end_encoding();

                let start = Instant::now();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                let elapsed = start.elapsed();

                println!(
                    "FFN up ({}x{}): {:?}",
                    hidden_size, intermediate_size, elapsed
                );
            }

            // Test FFN down
            {
                let ffn_input = forward_pool.get(intermediate_size * 4);
                let ffn_out = forward_pool.get(hidden_size * 4);

                let command_buffer = gpu_ctx.queue().new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                ctx.model.gemv_encode(
                    &encoder,
                    &ffn_input,
                    intermediate_size,
                    w2,
                    hidden_size,
                    &gpu_ctx,
                    &ffn_out,
                );
                encoder.end_encoding();

                let start = Instant::now();
                command_buffer.commit();
                command_buffer.wait_until_completed();
                let elapsed = start.elapsed();

                println!(
                    "FFN down ({}x{}): {:?}",
                    intermediate_size, hidden_size, elapsed
                );
            }

            println!("\n=== Summary ===");
            println!("Total kernels per layer: ~11");
            println!(
                "Total kernel time (estimated): {:.1} ms",
                28.0 * 11.0 * 0.1
            ); // Rough estimate
        })
        .join()
        .unwrap();
    }
}
