// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Detailed timing comparison: Our implementation vs llama.cpp

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");
    let tokens = tokenizer.encode("Hello");

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (ctx.gpu_ctx.as_ref(), ctx.gpu_kv_cache.as_mut())
        {
            let model = &ctx.model;
            let hidden_size = model.config.hidden_size;
            let n_layers = model.layers.len();
            let head_dim = model.config.head_dim;
            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            println!("\n=== Detailed Timing Analysis (Layer 0 Representative) ===\n");
            println!("Model: {}", model_path);
            println!("Layers: {}, Hidden size: {}, Head dim: {}", n_layers, hidden_size, head_dim);

            let hidden =
                model.embedding_lookup_impl(tokens[0]).expect("No embedding");
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            let layer = &model.layers[0];
            let layer_idx = 0;
            let pos = 0;

            let mut layer_input = hidden_buffer.clone();

            // Timing results storage
            let mut timing_results: Vec<(&str, std::time::Duration)> = Vec::new();

            // 1. Embedding lookup (CPU)
            let start = Instant::now();
            let _hidden = model.embedding_lookup_impl(tokens[0]).unwrap();
            timing_results.push(("Embedding lookup (CPU)", start.elapsed()));

            // 2. RMS Norm
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let attention_norm = layer.attention_norm.as_ref().unwrap();
            let norm_weight_buffer = gpu_ctx.get_or_create_buffer(attention_norm);
            let normed_buffer = forward_pool.get_zeroed(hidden_size * 4);

            gpu_ctx.rms_norm_encode(
                encoder,
                &layer_input,
                &norm_weight_buffer,
                &normed_buffer,
                1,
                hidden_size,
                1e-5,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("Attention RMS Norm", start.elapsed()));

            // 3. Q Projection
            let wq = layer.wq.as_ref().unwrap();
            let q_out_size = wq.shape[1];
            let n_heads = q_out_size / head_dim;

            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let q_buffer = forward_pool.get_zeroed(q_out_size * 4);

            model
                .gemv_encode(encoder, &normed_buffer, wq, q_out_size, gpu_ctx, &q_buffer)
                .unwrap();
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("Q projection", start.elapsed()));

            // 4. K Projection
            let wk = layer.wk.as_ref().unwrap();
            let k_out_size = wk.shape[1];
            let n_kv_heads = k_out_size / head_dim;

            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let k_buffer = forward_pool.get_zeroed(k_out_size * 4);

            model
                .gemv_encode(encoder, &normed_buffer, wk, k_out_size, gpu_ctx, &k_buffer)
                .unwrap();
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("K projection", start.elapsed()));

            // 5. V Projection
            let wv = layer.wv.as_ref().unwrap();
            let v_out_size = wv.shape[1];

            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let v_buffer = forward_pool.get_zeroed(v_out_size * 4);

            model
                .gemv_encode(encoder, &normed_buffer, wv, v_out_size, gpu_ctx, &v_buffer)
                .unwrap();
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("V projection", start.elapsed()));

            // 6. RoPE Q
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let q_rope_buffer = forward_pool.get_zeroed(q_out_size * 4);

            gpu_ctx.rope_neox_encode(
                encoder,
                &q_buffer,
                &[pos as i32],
                1,
                n_heads,
                head_dim,
                10000.0,
                1.0,
                &q_rope_buffer,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("RoPE (Q)", start.elapsed()));

            // 7. RoPE K
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let k_rope_buffer = forward_pool.get_zeroed(k_out_size * 4);

            gpu_ctx.rope_neox_encode(
                encoder,
                &k_buffer,
                &[pos as i32],
                1,
                n_kv_heads,
                head_dim,
                10000.0,
                1.0,
                &k_rope_buffer,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("RoPE (K)", start.elapsed()));

            // 8. KV Cache Update K
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let k_cache = kv_cache.k_buffer(layer_idx);

            gpu_ctx.kv_update_k_encode(
                encoder,
                &k_rope_buffer,
                k_cache,
                pos,
                kv_cache.max_seq(),
                n_kv_heads,
                head_dim,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("KV cache update (K)", start.elapsed()));

            // 9. KV Cache Update V
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let v_cache = kv_cache.v_buffer(layer_idx);

            gpu_ctx.kv_update_v_encode(
                encoder,
                &v_buffer,
                v_cache,
                pos,
                kv_cache.max_seq(),
                n_kv_heads,
                head_dim,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("KV cache update (V)", start.elapsed()));

            // 10. Flash Attention
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let attn_out_buffer = forward_pool.get_zeroed(q_out_size * 4);

            gpu_ctx.flash_attn_f16_encode(
                encoder,
                1,
                n_heads,
                n_kv_heads,
                head_dim,
                pos + 1,
                kv_cache.max_seq(),
                &q_rope_buffer,
                k_cache,
                v_cache,
                &attn_out_buffer,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("Flash Attention (decode)", start.elapsed()));

            // 11. Output Projection
            let wo = layer.wo.as_ref().unwrap();

            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let attn_proj_buffer = forward_pool.get_zeroed(hidden_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &attn_out_buffer,
                    wo,
                    hidden_size,
                    gpu_ctx,
                    &attn_proj_buffer,
                )
                .unwrap();
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("Output projection", start.elapsed()));

            // 12. Residual Add
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let residual_buffer = forward_pool.get_zeroed(hidden_size * 4);

            gpu_ctx.add_encode(
                encoder,
                &layer_input,
                &attn_proj_buffer,
                &residual_buffer,
                hidden_size,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("Residual add", start.elapsed()));

            // 13. FFN Norm
            let ffn_norm = layer.ffn_norm.as_ref().unwrap();

            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let ffn_norm_buffer = gpu_ctx.get_or_create_buffer(ffn_norm);
            let ffn_normed_buffer = forward_pool.get_zeroed(hidden_size * 4);

            gpu_ctx.rms_norm_encode(
                encoder,
                &residual_buffer,
                &ffn_norm_buffer,
                &ffn_normed_buffer,
                1,
                hidden_size,
                1e-5,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("FFN RMS Norm", start.elapsed()));

            // 14. Gate Projection
            let w1 = layer.w1.as_ref().unwrap();
            let intermediate_size = w1.shape[1];

            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let gate_buffer = forward_pool.get_zeroed(intermediate_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &ffn_normed_buffer,
                    w1,
                    intermediate_size,
                    gpu_ctx,
                    &gate_buffer,
                )
                .unwrap();
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("Gate projection", start.elapsed()));

            // 15. Up Projection
            let w3 = layer.w3.as_ref().unwrap();

            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let up_buffer = forward_pool.get_zeroed(intermediate_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &ffn_normed_buffer,
                    w3,
                    intermediate_size,
                    gpu_ctx,
                    &up_buffer,
                )
                .unwrap();
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("Up projection", start.elapsed()));

            // 16. SiLU Activation
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let silu_buffer = forward_pool.get_zeroed(intermediate_size * 4);

            gpu_ctx.silu_encode(encoder, &gate_buffer, &silu_buffer, intermediate_size);
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("SiLU activation", start.elapsed()));

            // 17. Element-wise Mul
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let hidden_ffn_buffer = forward_pool.get_zeroed(intermediate_size * 4);

            gpu_ctx.mul_encode(
                encoder,
                &silu_buffer,
                &up_buffer,
                &hidden_ffn_buffer,
                intermediate_size,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("Element-wise mul", start.elapsed()));

            // 18. Down Projection
            let w2 = layer.w2.as_ref().unwrap();

            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let ffn_out_buffer = forward_pool.get_zeroed(hidden_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &hidden_ffn_buffer,
                    w2,
                    hidden_size,
                    gpu_ctx,
                    &ffn_out_buffer,
                )
                .unwrap();
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("Down projection", start.elapsed()));

            // 19. Final Residual Add
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let final_buffer = forward_pool.get_zeroed(hidden_size * 4);

            gpu_ctx.add_encode(
                encoder,
                &residual_buffer,
                &ffn_out_buffer,
                &final_buffer,
                hidden_size,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            timing_results.push(("Final residual add", start.elapsed()));

            // Print results
            println!("\n=== Our Implementation Timing (Layer 0) ===\n");
            let mut total_time = std::time::Duration::ZERO;
            for (name, duration) in &timing_results {
                println!("{:40} {:8.3} ms", name, duration.as_secs_f64() * 1000.0);
                total_time += *duration;
            }
            println!("{:40} {:8.3} ms", "TOTAL (Layer 0)", total_time.as_secs_f64() * 1000.0);

            // Estimate full forward pass
            println!("\n=== Estimated Full Forward Pass ===\n");
            let estimated_total = total_time * n_layers as u32;
            println!("Estimated time for {} layers: {:.3} ms", n_layers, estimated_total.as_secs_f64() * 1000.0);
            println!("Estimated throughput: {:.1} t/s", 1000.0 / estimated_total.as_secs_f64() / 1000.0);
        }
    }
}
