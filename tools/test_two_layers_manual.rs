// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test manually encoding both layers

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

            let hidden =
                model.embedding_lookup_impl(9707).expect("No embedding");
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            println!("=== Manual encoding of both layers ===\n");

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // ==================== Layer 0 ====================
            let layer0 = &model.layers[0];
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

            let hidden_buffer = forward_pool.register(hidden_buffer);

            // Layer 0: RMS norm
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

            // Layer 0: QKV
            let q = forward_pool.get_zeroed(q_out_size * 4);
            let k = forward_pool.get_zeroed(k_out_size * 4);
            let v = forward_pool.get_zeroed(v_out_size * 4);
            model
                .gemv_encode(encoder, &normed, wq, q_out_size, gpu_ctx, &q)
                .unwrap();
            model
                .gemv_encode(encoder, &normed, wk, k_out_size, gpu_ctx, &k)
                .unwrap();
            model
                .gemv_encode(encoder, &normed, wv, v_out_size, gpu_ctx, &v)
                .unwrap();

            // Layer 0: RoPE
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

            // Layer 0: KV update
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

            use metal::ResourceRef;
            encoder.memory_barrier_with_resources(&[
                k_cache.as_ref() as &ResourceRef,
                v_cache.as_ref() as &ResourceRef,
            ]);

            // Layer 0: Flash attention
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

            // Layer 0: Output projection
            let attn_proj = forward_pool.get_zeroed(hidden_size * 4);
            model
                .gemv_encode(
                    encoder,
                    &attn_out,
                    wo,
                    hidden_size,
                    gpu_ctx,
                    &attn_proj,
                )
                .unwrap();

            // Layer 0: Residual
            let residual = forward_pool.get_zeroed(hidden_size * 4);
            gpu_ctx.add_encode(
                encoder,
                &hidden_buffer,
                &attn_proj,
                &residual,
                hidden_size,
            );

            // Layer 0: FFN norm
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

            // Layer 0: FFN
            let gate = forward_pool.get_zeroed(intermediate_size * 4);
            let up = forward_pool.get_zeroed(intermediate_size * 4);
            let silu = forward_pool.get_zeroed(intermediate_size * 4);
            let hidden_ffn = forward_pool.get_zeroed(intermediate_size * 4);
            let ffn_out = forward_pool.get_zeroed(hidden_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &ffn_normed,
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
                    w2,
                    hidden_size,
                    gpu_ctx,
                    &ffn_out,
                )
                .unwrap();

            // Layer 0: Final residual
            let layer0_out = forward_pool.get_zeroed(hidden_size * 4);
            gpu_ctx.add_encode(
                encoder,
                &residual,
                &ffn_out,
                &layer0_out,
                hidden_size,
            );

            println!("Layer 0 encoded");

            // ==================== Layer 1 ====================
            let layer1 = &model.layers[1];
            let attention_norm1 = layer1.attention_norm.as_ref().unwrap();
            let wq1 = layer1.wq.as_ref().unwrap();
            let wk1 = layer1.wk.as_ref().unwrap();
            let wv1 = layer1.wv.as_ref().unwrap();
            let wo1 = layer1.wo.as_ref().unwrap();
            let ffn_norm1 = layer1.ffn_norm.as_ref().unwrap();
            let w1_1 = layer1.w1.as_ref().unwrap();
            let w2_1 = layer1.w2.as_ref().unwrap();
            let w3_1 = layer1.w3.as_ref().unwrap();

            let q_out_size1 = wq1.shape[1];
            let k_out_size1 = wk1.shape[1];
            let v_out_size1 = wv1.shape[1];
            let intermediate_size1 = w1_1.shape[1];

            let layer1_input = forward_pool.register(layer0_out.clone());

            // Layer 1: RMS norm
            let normed1 = forward_pool.get_zeroed(hidden_size * 4);
            let norm_weight1 = gpu_ctx.get_or_create_buffer(attention_norm1);
            gpu_ctx.rms_norm_encode(
                encoder,
                &layer1_input,
                &norm_weight1,
                &normed1,
                1,
                hidden_size,
                1e-5,
            );

            // Layer 1: QKV
            let q1 = forward_pool.get_zeroed(q_out_size1 * 4);
            let k1 = forward_pool.get_zeroed(k_out_size1 * 4);
            let v1 = forward_pool.get_zeroed(v_out_size1 * 4);
            model
                .gemv_encode(encoder, &normed1, wq1, q_out_size1, gpu_ctx, &q1)
                .unwrap();
            model
                .gemv_encode(encoder, &normed1, wk1, k_out_size1, gpu_ctx, &k1)
                .unwrap();
            model
                .gemv_encode(encoder, &normed1, wv1, v_out_size1, gpu_ctx, &v1)
                .unwrap();

            // Layer 1: RoPE
            let q_rope1 = forward_pool.get_zeroed(q_out_size1 * 4);
            let k_rope1 = forward_pool.get_zeroed(k_out_size1 * 4);
            let actual_n_heads1 = q_out_size1 / head_dim;
            let actual_n_kv_heads1 = k_out_size1 / head_dim;

            gpu_ctx.rope_neox_encode(
                encoder,
                &q1,
                &[pos as i32],
                1,
                actual_n_heads1,
                head_dim,
                10000.0,
                1.0,
                &q_rope1,
            );
            gpu_ctx.rope_neox_encode(
                encoder,
                &k1,
                &[pos as i32],
                1,
                actual_n_kv_heads1,
                head_dim,
                10000.0,
                1.0,
                &k_rope1,
            );

            // Layer 1: KV update
            let k_cache1 = kv_cache.k_buffer(1);
            let v_cache1 = kv_cache.v_buffer(1);
            gpu_ctx.kv_update_k_encode(
                encoder,
                &k_rope1,
                k_cache1,
                pos,
                kv_cache.max_seq(),
                actual_n_kv_heads1,
                head_dim,
            );
            gpu_ctx.kv_update_v_encode(
                encoder,
                &v1,
                v_cache1,
                pos,
                kv_cache.max_seq(),
                actual_n_kv_heads1,
                head_dim,
            );

            encoder.memory_barrier_with_resources(&[
                k_cache1.as_ref() as &ResourceRef,
                v_cache1.as_ref() as &ResourceRef,
            ]);

            // Layer 1: Flash attention
            let attn_out1 = forward_pool.get_zeroed(q_out_size1 * 4);
            gpu_ctx.flash_attn_f16_encode(
                encoder,
                1,
                actual_n_heads1,
                actual_n_kv_heads1,
                head_dim,
                pos + 1,
                kv_cache.max_seq(),
                &q_rope1,
                k_cache1,
                v_cache1,
                &attn_out1,
            );

            // Layer 1: Output projection
            let attn_proj1 = forward_pool.get_zeroed(hidden_size * 4);
            model
                .gemv_encode(
                    encoder,
                    &attn_out1,
                    wo1,
                    hidden_size,
                    gpu_ctx,
                    &attn_proj1,
                )
                .unwrap();

            // Layer 1: Residual
            let residual1 = forward_pool.get_zeroed(hidden_size * 4);
            gpu_ctx.add_encode(
                encoder,
                &layer1_input,
                &attn_proj1,
                &residual1,
                hidden_size,
            );

            // Layer 1: FFN norm
            let ffn_normed1 = forward_pool.get_zeroed(hidden_size * 4);
            let ffn_norm_weight1 = gpu_ctx.get_or_create_buffer(ffn_norm1);
            gpu_ctx.rms_norm_encode(
                encoder,
                &residual1,
                &ffn_norm_weight1,
                &ffn_normed1,
                1,
                hidden_size,
                1e-5,
            );

            // Layer 1: FFN
            let gate1 = forward_pool.get_zeroed(intermediate_size1 * 4);
            let up1 = forward_pool.get_zeroed(intermediate_size1 * 4);
            let silu1 = forward_pool.get_zeroed(intermediate_size1 * 4);
            let hidden_ffn1 = forward_pool.get_zeroed(intermediate_size1 * 4);
            let ffn_out1 = forward_pool.get_zeroed(hidden_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &ffn_normed1,
                    w1_1,
                    intermediate_size1,
                    gpu_ctx,
                    &gate1,
                )
                .unwrap();
            model
                .gemv_encode(
                    encoder,
                    &ffn_normed1,
                    w3_1,
                    intermediate_size1,
                    gpu_ctx,
                    &up1,
                )
                .unwrap();

            gpu_ctx.silu_encode(encoder, &gate1, &silu1, intermediate_size1);
            gpu_ctx.mul_encode(
                encoder,
                &silu1,
                &up1,
                &hidden_ffn1,
                intermediate_size1,
            );
            model
                .gemv_encode(
                    encoder,
                    &hidden_ffn1,
                    w2_1,
                    hidden_size,
                    gpu_ctx,
                    &ffn_out1,
                )
                .unwrap();

            // Layer 1: Final residual
            let layer1_out = forward_pool.get_zeroed(hidden_size * 4);
            gpu_ctx.add_encode(
                encoder,
                &residual1,
                &ffn_out1,
                &layer1_out,
                hidden_size,
            );

            println!("Layer 1 encoded");

            // Execute
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Check results
            println!("\n=== Results ===\n");
            check_buffer("layer0_out", &layer0_out);
            check_buffer("layer1_out", &layer1_out);

            let k0 = kv_cache.get_k(0, 0);
            let k1 = kv_cache.get_k(1, 0);
            let k0_nan = k0.iter().filter(|&&v| v.is_nan()).count();
            let k1_nan = k1.iter().filter(|&&v| v.is_nan()).count();

            println!("\nK cache Layer 0: {} NaN", k0_nan);
            println!("K cache Layer 1: {} NaN", k1_nan);

            if k1_nan > 0 {
                println!("\n*** Layer 1 K cache still has NaN! ***");
            } else {
                println!("\n*** Manual encoding works! No NaN ***");
            }
        }
    }
}
