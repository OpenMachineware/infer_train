// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Trace Layer 0 NaN source by checking each intermediate buffer

use infer_train::inference::{Device, InferenceContext};
use std::cell::RefCell;
use std::rc::Rc;

fn check_buffer(name: &str, buffer: &metal::BufferRef) -> usize {
    let contents = buffer.contents() as *const f32;
    let len = buffer.length() as usize / 4;

    let num_nan =
        unsafe { (0..len).filter(|&i| (*contents.add(i)).is_nan()).count() };

    println!("{}: {} NaN", name, num_nan);
    num_nan
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (ctx.gpu_ctx.as_ref(), ctx.gpu_kv_cache.as_mut())
        {
            let model = &ctx.model;
            let hidden_size = model.config.hidden_size;
            let head_dim = model.config.head_dim;
            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            let layer0 = &model.layers[0];
            let hidden =
                model.embedding_lookup_impl(9707).expect("No embedding");
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            println!("=== Trace Layer 0 NaN source ===\n");

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Manually trace Layer 0 operations
            let attention_norm = layer0.attention_norm.as_ref().unwrap();
            let wq = layer0.wq.as_ref().unwrap();
            let wk = layer0.wk.as_ref().unwrap();
            let wv = layer0.wv.as_ref().unwrap();
            let wo = layer0.wo.as_ref().unwrap();
            let ffn_norm = layer0.ffn_norm.as_ref().unwrap();
            let w1 = layer0.w1.as_ref().unwrap();
            let w2 = layer0.w2.as_ref().unwrap();
            let w3 = layer0.w3.as_ref().unwrap();

            let q_out_size = wq.shape[1];
            let k_out_size = wk.shape[1];
            let v_out_size = wv.shape[1];
            let intermediate_size = w1.shape[1];

            // Register input
            let hidden_buffer = forward_pool.register(hidden_buffer);

            // 1. RMS norm
            let normed = forward_pool.get_zeroed(hidden_size * 4);
            let norm_weight = gpu_ctx.get_or_create_buffer(attention_norm);
            gpu_ctx.rms_norm_encode(
                encoder,
                &hidden_buffer,
                &norm_weight,
                &normed,
                1,
                hidden_size,
                1e-5,
            );

            // 2. QKV projections
            let q = forward_pool.get_zeroed(q_out_size * 4);
            let k = forward_pool.get_zeroed(k_out_size * 4);
            let v = forward_pool.get_zeroed(v_out_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &normed,
                    hidden_size,
                    wq,
                    q_out_size,
                    gpu_ctx,
                    &q,
                )
                .unwrap();
            model
                .gemv_encode(
                    encoder,
                    &normed,
                    hidden_size,
                    wk,
                    k_out_size,
                    gpu_ctx,
                    &k,
                )
                .unwrap();
            model
                .gemv_encode(
                    encoder,
                    &normed,
                    hidden_size,
                    wv,
                    v_out_size,
                    gpu_ctx,
                    &v,
                )
                .unwrap();

            // 3. RoPE
            let pos = kv_cache.seq_len();
            let q_rope = forward_pool.get_zeroed(q_out_size * 4);
            let k_rope = forward_pool.get_zeroed(k_out_size * 4);

            let actual_n_heads = q_out_size / head_dim;
            let actual_n_kv_heads = k_out_size / head_dim;

            gpu_ctx.rope_neox_encode(
                encoder,
                &q,
                &[pos as i32],
                1,
                actual_n_heads,
                head_dim,
                10000.0,
                1.0,
                &q_rope,
            );
            gpu_ctx.rope_neox_encode(
                encoder,
                &k,
                &[pos as i32],
                1,
                actual_n_kv_heads,
                head_dim,
                10000.0,
                1.0,
                &k_rope,
            );

            // 4. KV update
            let k_cache = kv_cache.k_buffer(0);
            let v_cache = kv_cache.v_buffer(0);
            gpu_ctx.kv_update_k_encode(
                encoder,
                &k_rope,
                k_cache,
                pos,
                kv_cache.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );
            gpu_ctx.kv_update_v_encode(
                encoder,
                &v,
                v_cache,
                pos,
                kv_cache.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );

            // Memory barrier for KV cache
            use metal::ResourceRef;
            encoder.memory_barrier_with_resources(&[
                k_cache.as_ref() as &ResourceRef,
                v_cache.as_ref() as &ResourceRef,
            ]);

            // 5. Flash attention
            let attn_out = forward_pool.get_zeroed(q_out_size * 4);
            gpu_ctx.flash_attn_f16_encode(
                encoder,
                1,
                actual_n_heads,
                actual_n_kv_heads,
                head_dim,
                pos + 1,
                kv_cache.max_seq(),
                &q_rope,
                k_cache,
                v_cache,
                &attn_out,
            );

            // 6. Output projection
            let attn_proj = forward_pool.get_zeroed(hidden_size * 4);
            model
                .gemv_encode(
                    encoder,
                    &attn_out,
                    q_out_size, // input size from attention output
                    wo,
                    hidden_size,
                    gpu_ctx,
                    &attn_proj,
                )
                .unwrap();

            // 7. Residual add
            let residual = forward_pool.get_zeroed(hidden_size * 4);
            gpu_ctx.add_encode(
                encoder,
                &hidden_buffer,
                &attn_proj,
                &residual,
                hidden_size,
            );

            // 8. FFN norm
            let ffn_normed = forward_pool.get_zeroed(hidden_size * 4);
            let ffn_norm_weight = gpu_ctx.get_or_create_buffer(ffn_norm);
            gpu_ctx.rms_norm_encode(
                encoder,
                &residual,
                &ffn_norm_weight,
                &ffn_normed,
                1,
                hidden_size,
                1e-5,
            );

            // 9. FFN
            let gate = forward_pool.get_zeroed(intermediate_size * 4);
            let up = forward_pool.get_zeroed(intermediate_size * 4);
            let silu = forward_pool.get_zeroed(intermediate_size * 4);
            let hidden_ffn = forward_pool.get_zeroed(intermediate_size * 4);
            let ffn_out = forward_pool.get_zeroed(hidden_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &ffn_normed,
                    hidden_size,
                    w1,
                    intermediate_size,
                    gpu_ctx,
                    &gate,
                )
                .unwrap();
            model
                .gemv_encode(
                    encoder,
                    &ffn_normed,
                    hidden_size,
                    w3,
                    intermediate_size,
                    gpu_ctx,
                    &up,
                )
                .unwrap();

            gpu_ctx.silu_encode(encoder, &gate, &silu, intermediate_size);
            gpu_ctx.mul_encode(
                encoder,
                &silu,
                &up,
                &hidden_ffn,
                intermediate_size,
            );
            model
                .gemv_encode(
                    encoder,
                    &hidden_ffn,
                    intermediate_size,
                    w2,
                    hidden_size,
                    gpu_ctx,
                    &ffn_out,
                )
                .unwrap();

            // 10. Final residual
            let final_out = forward_pool.get_zeroed(hidden_size * 4);
            gpu_ctx.add_encode(
                encoder,
                &residual,
                &ffn_out,
                &final_out,
                hidden_size,
            );

            // Execute
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Check all buffers
            println!("After GPU execution:\n");
            println!("--- Attention ---");
            check_buffer("normed", &normed);
            check_buffer("q", &q);
            check_buffer("k", &k);
            check_buffer("v", &v);
            check_buffer("q_rope", &q_rope);
            check_buffer("k_rope", &k_rope);
            check_buffer("attn_out", &attn_out);
            check_buffer("attn_proj", &attn_proj);
            check_buffer("residual", &residual);

            println!("\n--- FFN ---");
            check_buffer("ffn_normed", &ffn_normed);
            check_buffer("gate", &gate);
            check_buffer("up", &up);
            check_buffer("silu", &silu);
            check_buffer("hidden_ffn", &hidden_ffn);
            check_buffer("ffn_out", &ffn_out);
            check_buffer("final_out", &final_out);

            // Find first NaN
            println!("\n--- First NaN detection ---");
            let buffers: Vec<(&str, &metal::BufferRef)> = vec![
                ("normed", &normed),
                ("q", &q),
                ("k", &k),
                ("v", &v),
                ("q_rope", &q_rope),
                ("k_rope", &k_rope),
                ("attn_out", &attn_out),
                ("attn_proj", &attn_proj),
                ("residual", &residual),
                ("ffn_normed", &ffn_normed),
                ("gate", &gate),
                ("up", &up),
                ("silu", &silu),
                ("hidden_ffn", &hidden_ffn),
                ("ffn_out", &ffn_out),
                ("final_out", &final_out),
            ];

            for (name, buf) in buffers {
                let num_nan = check_buffer(name, buf);
                if num_nan > 0 {
                    println!("*** First NaN found in {} ***", name);
                    break;
                }
            }
        }
    }
}
