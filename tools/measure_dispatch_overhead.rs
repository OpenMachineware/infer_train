// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Measure dispatch overhead for separate vs single command buffer

use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

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

        // Measure dispatch overhead with a simple kernel
        println!("\n=== Dispatch Overhead Measurement ===\n");

        // Create a small buffer for testing
        let test_buf = device
            .new_buffer(1024, metal::MTLResourceOptions::StorageModeShared);

        // Test 1: Single command buffer
        let n_iterations = 100;
        let start = Instant::now();
        {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            // Multiple operations in single CB
            for _ in 0..n_iterations {
                gpu_ctx.add_encode(
                    enc,
                    &test_buf,
                    &test_buf,
                    &Rc::new(test_buf.clone()),
                    256,
                );
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let single_cb_time = start.elapsed().as_nanos() as f64 / 1000.0;
        println!(
            "Single command buffer ({} ops): {:.2} µs total, {:.2} µs/op",
            n_iterations,
            single_cb_time,
            single_cb_time / n_iterations as f64
        );

        // Test 2: Separate command buffers
        let start = Instant::now();
        for _ in 0..n_iterations {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            gpu_ctx.add_encode(
                enc,
                &test_buf,
                &test_buf,
                &Rc::new(test_buf.clone()),
                256,
            );
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let separate_cb_time = start.elapsed().as_nanos() as f64 / 1000.0;
        println!(
            "Separate command buffers ({} ops): {:.2} µs total, {:.2} µs/op",
            n_iterations,
            separate_cb_time,
            separate_cb_time / n_iterations as f64
        );

        println!("\nOverhead ratio: {:.1}x", separate_cb_time / single_cb_time);

        // Estimate actual overhead in layer execution
        let ops_per_layer = 14; // From profile
        let estimated_overhead_per_layer = (separate_cb_time - single_cb_time)
            / n_iterations as f64
            * ops_per_layer as f64;
        println!(
            "\nEstimated dispatch overhead per layer: {:.2} µs",
            estimated_overhead_per_layer
        );
        println!(
            "This accounts for {:.1}% of the 9117 µs layer time",
            estimated_overhead_per_layer / 9117.0 * 100.0
        );

        // Now test actual layer 0 with single command buffer (if possible)
        println!("\n=== Layer 0 with Minimal Command Buffers ===\n");

        let reuse_pool = Rc::new(RefCell::new(Vec::new()));
        let mut pool = infer_train::buffer::ForwardBufferPool::new(
            device.clone(),
            reuse_pool,
        );

        let layer = &model.layers[0];
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
        let pos = kv.seq_len();

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

        // Attention block (one command buffer)
        let start = Instant::now();
        {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();

            // RMS Norm
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

            // QKV projections
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

            // RoPE
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

            // KV cache update
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

            // Flash Attention
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

            // Output projection
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

            // Residual
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
        }
        let attention_time = start.elapsed().as_nanos() as f64 / 1000.0;
        println!("Attention block (single CB): {:.2} µs", attention_time);

        // FFN block (one command buffer)
        let start = Instant::now();
        {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();

            // FFN norm
            let ffn_norm = layer.ffn_norm.as_ref().unwrap();
            let ffn_norm_buffer = gpu_ctx.get_or_create_buffer(ffn_norm);
            gpu_ctx.rms_norm_encode(
                enc,
                &residual_buffer,
                &ffn_norm_buffer,
                &ffn_normed_buffer,
                1,
                hidden_size,
                1e-5,
            );

            // Gate + Up
            let w1 = layer.w1.as_ref().unwrap();
            let w3 = layer.w3.as_ref().unwrap();
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

            // SiLU + Mul
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

            // Down projection
            let w2 = layer.w2.as_ref().unwrap();
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

            // Final residual
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
        }
        let ffn_time = start.elapsed().as_nanos() as f64 / 1000.0;
        println!("FFN block (single CB): {:.2} µs", ffn_time);

        let total_with_2_cb = attention_time + ffn_time;
        println!(
            "\nLayer 0 total (2 command buffers): {:.2} µs",
            total_with_2_cb
        );
        println!(
            "Savings vs 14 CBs: {:.2} µs ({:.1}%)",
            9117.0 - total_with_2_cb,
            (9117.0 - total_with_2_cb) / 9117.0 * 100.0
        );

        // Estimate full model time
        let n_layers = model.layers.len();
        let estimated_full_model = total_with_2_cb * n_layers as f64 / 1000.0;
        println!(
            "\nEstimated full model ({} layers, 2 CB/layer): {:.2} ms",
            n_layers, estimated_full_model
        );
        println!("Target (llama.cpp): ~0.29 ms");
        println!("Gap: {:.1}x", estimated_full_model / 0.29);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
