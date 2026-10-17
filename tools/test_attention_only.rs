// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test attention only to isolate NaN issue

use infer_train::inference::{Device, InferenceContext};
use std::cell::RefCell;
use std::rc::Rc;

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

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let layer0 = &model.layers[0];
            let head_dim = model.config.head_dim;

            // Attention norm
            let norm_weight_buffer = gpu_ctx
                .get_or_create_buffer(layer0.attention_norm.as_ref().unwrap());
            let normed_buffer = forward_pool.get_zeroed(hidden_size * 4);
            gpu_ctx.rms_norm_encode(
                encoder,
                &hidden_buffer,
                &norm_weight_buffer,
                &normed_buffer,
                1,
                hidden_size,
                1e-5,
            );

            // QKV projections
            let wq = layer0.wq.as_ref().expect("No wq");
            let wk = layer0.wk.as_ref().expect("No wk");
            let wv = layer0.wv.as_ref().expect("No wv");

            let q_out_size = wq.shape[1];
            let k_out_size = wk.shape[1];
            let v_out_size = wv.shape[1];
            let actual_n_heads = q_out_size / head_dim;
            let actual_n_kv_heads = k_out_size / head_dim;

            let q_buffer = forward_pool.get_zeroed(q_out_size * 4);
            let k_buffer = forward_pool.get_zeroed(k_out_size * 4);
            let v_buffer = forward_pool.get_zeroed(v_out_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &normed_buffer,
                    wq,
                    q_out_size,
                    gpu_ctx,
                    &q_buffer,
                )
                .expect("Q failed");
            model
                .gemv_encode(
                    encoder,
                    &normed_buffer,
                    wk,
                    k_out_size,
                    gpu_ctx,
                    &k_buffer,
                )
                .expect("K failed");
            model
                .gemv_encode(
                    encoder,
                    &normed_buffer,
                    wv,
                    v_out_size,
                    gpu_ctx,
                    &v_buffer,
                )
                .expect("V failed");

            // RoPE
            let q_rope_buffer = forward_pool.get_zeroed(q_out_size * 4);
            let k_rope_buffer = forward_pool.get_zeroed(k_out_size * 4);

            gpu_ctx.rope_neox_encode(
                encoder,
                &q_buffer,
                &[0],
                1,
                actual_n_heads,
                head_dim,
                10000.0,
                1.0,
                &q_rope_buffer,
            );
            gpu_ctx.rope_neox_encode(
                encoder,
                &k_buffer,
                &[0],
                1,
                actual_n_kv_heads,
                head_dim,
                10000.0,
                1.0,
                &k_rope_buffer,
            );

            // KV update
            let k_cache = kv_cache.k_buffer(0);
            let v_cache = kv_cache.v_buffer(0);
            gpu_ctx.kv_update_k_encode(
                encoder,
                &k_rope_buffer,
                k_cache,
                0,
                kv_cache.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );
            gpu_ctx.kv_update_v_encode(
                encoder,
                &v_buffer,
                v_cache,
                0,
                kv_cache.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );

            // Memory barrier
            use metal::ResourceRef;
            encoder.memory_barrier_with_resources(&[
                k_cache.as_ref() as &ResourceRef,
                v_cache.as_ref() as &ResourceRef,
            ]);

            // Flash attention
            let attn_out_buffer = forward_pool.get_zeroed(q_out_size * 4);
            gpu_ctx.flash_attn_f16_encode(
                encoder,
                1,
                actual_n_heads,
                actual_n_kv_heads,
                head_dim,
                1,
                kv_cache.max_seq(),
                &q_rope_buffer,
                k_cache,
                v_cache,
                &attn_out_buffer,
            );

            // WO projection
            let wo = layer0.wo.as_ref().expect("No wo");
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
                .expect("WO failed");

            // Residual add
            let residual_buffer = forward_pool.get_zeroed(hidden_size * 4);
            gpu_ctx.add_encode(
                encoder,
                &hidden_buffer,
                &attn_proj_buffer,
                &residual_buffer,
                hidden_size,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Check results
            let mut q_data = vec![0.0f32; q_out_size];
            let mut k_data = vec![0.0f32; k_out_size];
            let mut attn_out_data = vec![0.0f32; q_out_size];
            let mut attn_proj_data = vec![0.0f32; hidden_size];
            let mut residual_data = vec![0.0f32; hidden_size];

            unsafe {
                std::ptr::copy_nonoverlapping(
                    q_buffer.contents() as *const f32,
                    q_data.as_mut_ptr(),
                    q_out_size,
                );
                std::ptr::copy_nonoverlapping(
                    k_buffer.contents() as *const f32,
                    k_data.as_mut_ptr(),
                    k_out_size,
                );
                std::ptr::copy_nonoverlapping(
                    attn_out_buffer.contents() as *const f32,
                    attn_out_data.as_mut_ptr(),
                    q_out_size,
                );
                std::ptr::copy_nonoverlapping(
                    attn_proj_buffer.contents() as *const f32,
                    attn_proj_data.as_mut_ptr(),
                    hidden_size,
                );
                std::ptr::copy_nonoverlapping(
                    residual_buffer.contents() as *const f32,
                    residual_data.as_mut_ptr(),
                    hidden_size,
                );
            }

            println!(
                "Q: {} NaN, first 5: {:?}",
                q_data.iter().filter(|&&v| v.is_nan()).count(),
                &q_data[..5.min(q_data.len())]
            );
            println!(
                "K: {} NaN, first 5: {:?}",
                k_data.iter().filter(|&&v| v.is_nan()).count(),
                &k_data[..5.min(k_data.len())]
            );
            println!(
                "Attn out: {} NaN, first 5: {:?}",
                attn_out_data.iter().filter(|&&v| v.is_nan()).count(),
                &attn_out_data[..5.min(attn_out_data.len())]
            );
            println!(
                "Attn proj: {} NaN, first 5: {:?}",
                attn_proj_data.iter().filter(|&&v| v.is_nan()).count(),
                &attn_proj_data[..5.min(attn_proj_data.len())]
            );
            println!(
                "Residual: {} NaN, first 5: {:?}",
                residual_data.iter().filter(|&&v| v.is_nan()).count(),
                &residual_data[..5.min(residual_data.len())]
            );

            let nan_positions: Vec<usize> = residual_data
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_nan())
                .map(|(i, _)| i)
                .collect();
            println!(
                "Residual NaN positions: {:?}",
                &nan_positions[..10.min(nan_positions.len())]
            );
        }
    }
}
