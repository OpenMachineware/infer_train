// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile: Detailed timing for each kernel in GPU forward pass

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
use std::cell::RefCell;
use std::rc::Rc;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use metal::MTLResourceOptions;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");
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
            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            println!("\n=== Detailed Kernel Timing Profile ===");
            println!("Layers: {}", n_layers);
            println!("Hidden size: {}", hidden_size);

            // Test with single token
            let hidden =
                model.embedding_lookup_impl(tokens[0]).expect("No embedding");
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            // Profile each operation in Layer 0 only (representative)
            println!("\n=== Layer 0 Detailed Timing ===\n");

            let layer = &model.layers[0];
            let layer_idx = 0;
            let pos = 0;

            // Create separate command buffers for each operation to measure timing
            let mut layer_input = hidden_buffer.clone();

            // 1. RMS Norm
            let norm_start = std::time::Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();

            let attention_norm = layer.attention_norm.as_ref().unwrap();
            let norm_weight_buffer =
                gpu_ctx.get_or_create_buffer(attention_norm);
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
            let norm_time = norm_start.elapsed();

            println!("1. RMS Norm: {:?}", norm_time);

            // 2. Q Projection
            let wq = layer.wq.as_ref().unwrap();
            let q_out_size = wq.shape[1];

            let q_start = std::time::Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let q_buffer = forward_pool.get_zeroed(q_out_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &normed_buffer,
                    wq,
                    q_out_size,
                    gpu_ctx,
                    &q_buffer,
                )
                .unwrap();
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let q_time = q_start.elapsed();
            println!(
                "2. Q Projection ({}x{}): {:?}",
                hidden_size, q_out_size, q_time
            );

            // 3. K Projection
            let wk = layer.wk.as_ref().unwrap();
            let k_out_size = wk.shape[1];

            let k_start = std::time::Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let k_buffer = forward_pool.get_zeroed(k_out_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &normed_buffer,
                    wk,
                    k_out_size,
                    gpu_ctx,
                    &k_buffer,
                )
                .unwrap();
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let k_time = k_start.elapsed();
            println!(
                "3. K Projection ({}x{}): {:?}",
                hidden_size, k_out_size, k_time
            );

            // 4. V Projection
            let wv = layer.wv.as_ref().unwrap();
            let v_out_size = wv.shape[1];

            let v_start = std::time::Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let v_buffer = forward_pool.get_zeroed(v_out_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &normed_buffer,
                    wv,
                    v_out_size,
                    gpu_ctx,
                    &v_buffer,
                )
                .unwrap();
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let v_time = v_start.elapsed();
            println!(
                "4. V Projection ({}x{}): {:?}",
                hidden_size, v_out_size, v_time
            );

            // 5. RoPE for Q
            let head_dim = model.config.head_dim;
            let n_heads = q_out_size / head_dim;

            let rope_q_start = std::time::Instant::now();
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
            let rope_q_time = rope_q_start.elapsed();
            println!("5. RoPE Q: {:?}", rope_q_time);

            // 6. RoPE for K
            let n_kv_heads = k_out_size / head_dim;

            let rope_k_start = std::time::Instant::now();
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
            let rope_k_time = rope_k_start.elapsed();
            println!("6. RoPE K: {:?}", rope_k_time);

            // 7. KV Cache Update K
            let kv_update_k_start = std::time::Instant::now();
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
            let kv_update_k_time = kv_update_k_start.elapsed();
            println!("7. KV Cache Update K: {:?}", kv_update_k_time);

            // 8. KV Cache Update V
            let kv_update_v_start = std::time::Instant::now();
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
            let kv_update_v_time = kv_update_v_start.elapsed();
            println!("8. KV Cache Update V: {:?}", kv_update_v_time);

            // 9. Flash Attention
            let attn_start = std::time::Instant::now();
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
            let attn_time = attn_start.elapsed();
            println!("9. Flash Attention (seq_len=1): {:?}", attn_time);

            // 10. Output Projection
            let wo = layer.wo.as_ref().unwrap();

            let wo_start = std::time::Instant::now();
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
            let wo_time = wo_start.elapsed();
            println!("10. Output Projection: {:?}", wo_time);

            // 11. Residual Add
            let add_start = std::time::Instant::now();
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
            let add_time = add_start.elapsed();
            println!("11. Residual Add: {:?}", add_time);

            // 12. FFN Norm
            let ffn_norm_start = std::time::Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let ffn_norm = layer.ffn_norm.as_ref().unwrap();
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
            let ffn_norm_time = ffn_norm_start.elapsed();
            println!("12. FFN Norm: {:?}", ffn_norm_time);

            // 13. Gate Projection
            let w1 = layer.w1.as_ref().unwrap();
            let intermediate_size = w1.shape[1];

            let gate_start = std::time::Instant::now();
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
            let gate_time = gate_start.elapsed();
            println!(
                "13. Gate Projection ({}x{}): {:?}",
                hidden_size, intermediate_size, gate_time
            );

            // 14. Up Projection
            let w3 = layer.w3.as_ref().unwrap();

            let up_start = std::time::Instant::now();
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
            let up_time = up_start.elapsed();
            println!(
                "14. Up Projection ({}x{}): {:?}",
                hidden_size, intermediate_size, up_time
            );

            // 15. SiLU
            let silu_start = std::time::Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let silu_buffer = forward_pool.get_zeroed(intermediate_size * 4);

            gpu_ctx.silu_encode(
                encoder,
                &gate_buffer,
                &silu_buffer,
                intermediate_size,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let silu_time = silu_start.elapsed();
            println!("15. SiLU: {:?}", silu_time);

            // 16. Mul
            let mul_start = std::time::Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            let hidden_ffn_buffer =
                forward_pool.get_zeroed(intermediate_size * 4);

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
            let mul_time = mul_start.elapsed();
            println!("16. Mul: {:?}", mul_time);

            // 17. Down Projection
            let w2 = layer.w2.as_ref().unwrap();

            let down_start = std::time::Instant::now();
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
            let down_time = down_start.elapsed();
            println!(
                "17. Down Projection ({}x{}): {:?}",
                intermediate_size, hidden_size, down_time
            );

            // 18. Final Residual Add
            let final_add_start = std::time::Instant::now();
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
            let final_add_time = final_add_start.elapsed();
            println!("18. Final Residual Add: {:?}", final_add_time);

            // Summary
            let total_time = norm_time
                + q_time
                + k_time
                + v_time
                + rope_q_time
                + rope_k_time
                + kv_update_k_time
                + kv_update_v_time
                + attn_time
                + wo_time
                + add_time
                + ffn_norm_time
                + gate_time
                + up_time
                + silu_time
                + mul_time
                + down_time
                + final_add_time;

            println!("\n=== Layer 0 Total: {:?} ===", total_time);
            println!("\n=== Breakdown ===");
            println!(
                "GEMV operations: {:?}",
                q_time
                    + k_time
                    + v_time
                    + wo_time
                    + gate_time
                    + up_time
                    + down_time
            );
            println!("Flash Attention: {:?}", attn_time);
            println!(
                "Other kernels: {:?}",
                total_time
                    - q_time
                    - k_time
                    - v_time
                    - wo_time
                    - gate_time
                    - up_time
                    - down_time
                    - attn_time
            );
        }
    }
}
