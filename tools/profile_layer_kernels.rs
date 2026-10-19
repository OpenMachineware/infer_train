// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile individual kernel performance in layer 0

use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

fn time_kernel<F: FnOnce()>(name: &str, f: F) -> f64 {
    let start = Instant::now();
    f();
    let elapsed = start.elapsed();
    let us = elapsed.as_nanos() as f64 / 1000.0;
    println!("  {}: {:.2} µs", name, us);
    us
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

        let mut kv = KVCacheGPU::new(device, config, 2048);
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

        println!("\n=== Layer 0 Kernel Performance Profile ===\n");

        let layer = &model.layers[0];
        let pos = kv.seq_len();
        let wq = layer.wq.as_ref().unwrap();
        let wk = layer.wk.as_ref().unwrap();
        let wv = layer.wv.as_ref().unwrap();
        let wo = layer.wo.as_ref().unwrap();
        let q_out_size = wq.shape[1];
        let k_out_size = wk.shape[1];
        let v_out_size = wv.shape[1];
        let actual_n_heads = q_out_size / head_dim;
        let actual_n_kv_heads = k_out_size / head_dim;
        let intermediate_size =
            layer.w1.as_ref().map(|w| w.shape[1]).unwrap_or(0);

        // Create buffers
        let hidden_buffer = pool.register(emb_buf.clone());
        let normed_buffer = pool.get_zeroed(hidden_size * 4);
        let q_buffer = pool.get_zeroed(q_out_size * 4);
        let k_buffer = pool.get_zeroed(k_out_size * 4);
        let v_buffer = pool.get_zeroed(v_out_size * 4);
        let q_rope_buffer = pool.get_zeroed(q_out_size * 4);
        let k_rope_buffer = pool.get_zeroed(k_out_size * 4);
        let attn_out_buffer = pool.get_zeroed(q_out_size * 4);
        let attn_proj_buffer = pool.get_zeroed(hidden_size * 4);
        let residual_buffer = pool.get_zeroed(hidden_size * 4);
        let ffn_normed_buffer = pool.get_zeroed(hidden_size * 4);
        let gate_buffer = pool.get_zeroed(intermediate_size * 4);
        let up_buffer = pool.get_zeroed(intermediate_size * 4);
        let silu_buffer = pool.get_zeroed(intermediate_size * 4);
        let hidden_ffn_buffer = pool.get_zeroed(intermediate_size * 4);
        let ffn_out_buffer = pool.get_zeroed(hidden_size * 4);
        let final_buffer = pool.get_zeroed(hidden_size * 4);

        let attention_norm = layer.attention_norm.as_ref().unwrap();
        let norm_weight_buffer = gpu_ctx.get_or_create_buffer(attention_norm);

        let mut total_time = 0.0;

        // Step 1: RMS Norm
        total_time += time_kernel("RMS Norm", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            gpu_ctx.rms_norm_encode(
                enc,
                &hidden_buffer,
                &norm_weight_buffer,
                &normed_buffer,
                1,
                hidden_size,
                1e-5,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Step 2: Q projection
        total_time += time_kernel("Q projection (GEMV)", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
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
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Step 3: K projection
        total_time += time_kernel("K projection (GEMV)", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
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
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Step 4: V projection
        total_time += time_kernel("V projection (GEMV)", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
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
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Step 5: RoPE
        total_time += time_kernel("RoPE (Q+K)", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
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
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Step 6: KV cache update
        total_time += time_kernel("KV cache update", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
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
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Step 7: Flash Attention
        total_time += time_kernel("Flash Attention", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            let k_cache = kv.k_buffer(0);
            let v_cache = kv.v_buffer(0);
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
        });

        // Step 8: Output projection
        total_time += time_kernel("Output projection (GEMV)", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            model
                .gemv_encode(
                    enc,
                    &attn_out_buffer,
                    wo,
                    hidden_size,
                    gpu_ctx,
                    &attn_proj_buffer,
                )
                .unwrap();
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Step 9: Residual add
        total_time += time_kernel("Residual add", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            gpu_ctx.add_encode(
                enc,
                &hidden_buffer,
                &attn_proj_buffer,
                &residual_buffer,
                hidden_size,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Step 10: FFN norm
        let ffn_norm = layer.ffn_norm.as_ref().unwrap();
        let ffn_norm_buffer = gpu_ctx.get_or_create_buffer(ffn_norm);
        total_time += time_kernel("FFN norm", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            gpu_ctx.rms_norm_encode(
                enc,
                &residual_buffer,
                &ffn_norm_buffer,
                &ffn_normed_buffer,
                1,
                hidden_size,
                1e-5,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Step 11: Gate + Up projections
        let w1 = layer.w1.as_ref().unwrap();
        let w3 = layer.w3.as_ref().unwrap();
        total_time += time_kernel("Gate + Up (GEMV×2)", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            model
                .gemv_encode(
                    enc,
                    &ffn_normed_buffer,
                    w1,
                    intermediate_size,
                    gpu_ctx,
                    &gate_buffer,
                )
                .unwrap();
            model
                .gemv_encode(
                    enc,
                    &ffn_normed_buffer,
                    w3,
                    intermediate_size,
                    gpu_ctx,
                    &up_buffer,
                )
                .unwrap();
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Step 12: SiLU * Up
        total_time += time_kernel("SiLU + Mul", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            gpu_ctx.silu_encode(
                enc,
                &gate_buffer,
                &silu_buffer,
                intermediate_size,
            );
            gpu_ctx.mul_encode(
                enc,
                &silu_buffer,
                &up_buffer,
                &hidden_ffn_buffer,
                intermediate_size,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Step 13: Down projection
        let w2 = layer.w2.as_ref().unwrap();
        total_time += time_kernel("Down projection (GEMV)", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            model
                .gemv_encode(
                    enc,
                    &hidden_ffn_buffer,
                    w2,
                    hidden_size,
                    gpu_ctx,
                    &ffn_out_buffer,
                )
                .unwrap();
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        // Step 14: Final residual add
        total_time += time_kernel("Final residual", || {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            gpu_ctx.add_encode(
                enc,
                &residual_buffer,
                &ffn_out_buffer,
                &final_buffer,
                hidden_size,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        println!("\n=== Summary ===");
        println!("Total layer 0 time: {:.2} µs", total_time);
        println!("GEMV ops: 6 (Q, K, V, Output, Gate+Up, Down)");
        println!(
            "Other ops: 8 (Norm, RoPE, KV update, Flash Attn, Add, FFN norm, SiLU+Mul)"
        );

        // Estimate full model time
        let n_layers = model.layers.len();
        let estimated_total = total_time * n_layers as f64 / 1000.0; // ms
        println!(
            "\nEstimated full model time ({} layers): {:.2} ms",
            n_layers, estimated_total
        );

        // Compare with llama.cpp
        println!("\nTarget (llama.cpp): ~0.29 ms per token for Qwen3-0.6B");
        println!("Current gap: {:.1}x slower", estimated_total / 0.29);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
