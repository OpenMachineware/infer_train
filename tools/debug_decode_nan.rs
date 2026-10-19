// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug NaN in decode (position > 0)

use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
use std::cell::RefCell;
use std::rc::Rc;

fn check_buffer(name: &str, buf: &metal::BufferRef, size: usize) -> usize {
    let mut data = vec![0.0f32; size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents() as *const f32,
            data.as_mut_ptr(),
            size,
        );
    }
    let nan_count = data.iter().filter(|&&v| v.is_nan()).count();
    if nan_count > 0 {
        println!("{}: {} NaN / {}", name, nan_count, size);
    } else {
        println!("{}: OK (first 3: {:?})", name, &data[..3.min(size)]);
    }
    nan_count
}

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let model = &ctx.model;
        let gpu_ctx = ctx.gpu_ctx.as_ref().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();
        let config = &model.config;
        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim;

        // Create fresh KV cache
        let mut kv = KVCacheGPU::new(device, config, 2048);

        println!("\n=== Prefill (Position 0) ===");
        let emb = model.embedding_lookup_impl(1).unwrap();
        let emb_buf = Rc::new(device.new_buffer_with_data(
            emb.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let reuse_pool = Rc::new(RefCell::new(Vec::new()));
        let mut pool = infer_train::buffer::ForwardBufferPool::new(
            device.clone(),
            reuse_pool,
        );

        // Run layer 0 at position 0
        let layer = &model.layers[0];
        let pos = kv.seq_len();
        println!("Position: {}", pos);

        let wq = layer.wq.as_ref().unwrap();
        let wk = layer.wk.as_ref().unwrap();
        let wv = layer.wv.as_ref().unwrap();
        let q_out_size = wq.shape[1];
        let k_out_size = wk.shape[1];
        let v_out_size = wv.shape[1];
        let actual_n_heads = q_out_size / head_dim;
        let actual_n_kv_heads = k_out_size / head_dim;

        // Create buffers
        let hidden_buffer = pool.register(emb_buf);
        let normed_buffer = pool.get_zeroed(hidden_size * 4);
        let q_buffer = pool.get_zeroed(q_out_size * 4);
        let k_buffer = pool.get_zeroed(k_out_size * 4);
        let v_buffer = pool.get_zeroed(v_out_size * 4);
        let q_rope_buffer = pool.get_zeroed(q_out_size * 4);
        let k_rope_buffer = pool.get_zeroed(k_out_size * 4);
        let attn_out_buffer = pool.get_zeroed(q_out_size * 4);

        // Attention
        {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();

            let attention_norm = layer.attention_norm.as_ref().unwrap();
            let norm_weight_buffer =
                gpu_ctx.get_or_create_buffer(attention_norm);
            gpu_ctx.rms_norm_encode(
                enc,
                &hidden_buffer,
                &norm_weight_buffer,
                &normed_buffer,
                1,
                hidden_size,
                1e-5,
            );
            model
                .gemv_encode(
                    enc,
                    &normed_buffer,
                    wq,
                    q_out_size,
                    gpu_ctx,
                    &q_buffer,
                )
                .unwrap();
            model
                .gemv_encode(
                    enc,
                    &normed_buffer,
                    wk,
                    k_out_size,
                    gpu_ctx,
                    &k_buffer,
                )
                .unwrap();
            model
                .gemv_encode(
                    enc,
                    &normed_buffer,
                    wv,
                    v_out_size,
                    gpu_ctx,
                    &v_buffer,
                )
                .unwrap();

            gpu_ctx.rope_neox_encode(
                enc,
                &q_buffer,
                &[pos as i32],
                1,
                actual_n_heads,
                head_dim,
                10000.0,
                1.0,
                &q_rope_buffer,
            );
            gpu_ctx.rope_neox_encode(
                enc,
                &k_buffer,
                &[pos as i32],
                1,
                actual_n_kv_heads,
                head_dim,
                10000.0,
                1.0,
                &k_rope_buffer,
            );

            let k_cache = kv.k_buffer(0);
            let v_cache = kv.v_buffer(0);
            gpu_ctx.kv_update_k_encode(
                enc,
                &k_rope_buffer,
                k_cache,
                pos,
                kv.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );
            gpu_ctx.kv_update_v_encode(
                enc,
                &v_buffer,
                v_cache,
                pos,
                kv.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );

            gpu_ctx.flash_attn_f16_encode(
                enc,
                1,
                actual_n_heads,
                actual_n_kv_heads,
                head_dim,
                pos + 1,
                kv.max_seq(),
                &q_rope_buffer,
                k_cache,
                v_cache,
                &attn_out_buffer,
            );

            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        check_buffer("Q after RoPE", &q_rope_buffer, q_out_size);
        check_buffer("K after RoPE", &k_rope_buffer, k_out_size);
        check_buffer(
            "KV cache K",
            kv.k_buffer(0),
            actual_n_kv_heads * head_dim * kv.max_seq(),
        );
        check_buffer(
            "KV cache V",
            kv.v_buffer(0),
            actual_n_kv_heads * head_dim * kv.max_seq(),
        );
        check_buffer("Attention output", &attn_out_buffer, q_out_size);

        kv.advance();
        println!("\nKV cache advanced to position: {}", kv.seq_len());

        println!("\n=== Decode (Position 1) ===");
        let emb2 = model.embedding_lookup_impl(2).unwrap();
        let emb_buf2 = Rc::new(device.new_buffer_with_data(
            emb2.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Clear pool
        pool.release_all();

        let hidden_buffer2 = pool.register(emb_buf2);
        let normed_buffer2 = pool.get_zeroed(hidden_size * 4);
        let q_buffer2 = pool.get_zeroed(q_out_size * 4);
        let k_buffer2 = pool.get_zeroed(k_out_size * 4);
        let v_buffer2 = pool.get_zeroed(v_out_size * 4);
        let q_rope_buffer2 = pool.get_zeroed(q_out_size * 4);
        let k_rope_buffer2 = pool.get_zeroed(k_out_size * 4);
        let attn_out_buffer2 = pool.get_zeroed(q_out_size * 4);

        let pos2 = kv.seq_len();
        println!("Position: {}", pos2);

        // Attention at position 1
        {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();

            let attention_norm = layer.attention_norm.as_ref().unwrap();
            let norm_weight_buffer =
                gpu_ctx.get_or_create_buffer(attention_norm);
            gpu_ctx.rms_norm_encode(
                enc,
                &hidden_buffer2,
                &norm_weight_buffer,
                &normed_buffer2,
                1,
                hidden_size,
                1e-5,
            );
            model
                .gemv_encode(
                    enc,
                    &normed_buffer2,
                    wq,
                    q_out_size,
                    gpu_ctx,
                    &q_buffer2,
                )
                .unwrap();
            model
                .gemv_encode(
                    enc,
                    &normed_buffer2,
                    wk,
                    k_out_size,
                    gpu_ctx,
                    &k_buffer2,
                )
                .unwrap();
            model
                .gemv_encode(
                    enc,
                    &normed_buffer2,
                    wv,
                    v_out_size,
                    gpu_ctx,
                    &v_buffer2,
                )
                .unwrap();

            gpu_ctx.rope_neox_encode(
                enc,
                &q_buffer2,
                &[pos2 as i32],
                1,
                actual_n_heads,
                head_dim,
                10000.0,
                1.0,
                &q_rope_buffer2,
            );
            gpu_ctx.rope_neox_encode(
                enc,
                &k_buffer2,
                &[pos2 as i32],
                1,
                actual_n_kv_heads,
                head_dim,
                10000.0,
                1.0,
                &k_rope_buffer2,
            );

            let k_cache = kv.k_buffer(0);
            let v_cache = kv.v_buffer(0);
            gpu_ctx.kv_update_k_encode(
                enc,
                &k_rope_buffer2,
                k_cache,
                pos2,
                kv.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );
            gpu_ctx.kv_update_v_encode(
                enc,
                &v_buffer2,
                v_cache,
                pos2,
                kv.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );

            gpu_ctx.flash_attn_f16_encode(
                enc,
                1,
                actual_n_heads,
                actual_n_kv_heads,
                head_dim,
                pos2 + 1,
                kv.max_seq(),
                &q_rope_buffer2,
                k_cache,
                v_cache,
                &attn_out_buffer2,
            );

            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        check_buffer("Q after RoPE (pos 1)", &q_rope_buffer2, q_out_size);
        check_buffer("K after RoPE (pos 1)", &k_rope_buffer2, k_out_size);
        check_buffer(
            "KV cache K (pos 1)",
            kv.k_buffer(0),
            actual_n_kv_heads * head_dim * kv.max_seq(),
        );
        check_buffer(
            "KV cache V (pos 1)",
            kv.v_buffer(0),
            actual_n_kv_heads * head_dim * kv.max_seq(),
        );
        check_buffer("Attention output (pos 1)", &attn_out_buffer2, q_out_size);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
