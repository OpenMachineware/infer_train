// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Trace each operation in Layer 0 and Layer 1

use infer_train::inference::{Device, InferenceContext};
use std::cell::RefCell;
use std::rc::Rc;

fn check_buffer(buf: &metal::Buffer, name: &str, size: usize) {
    let data = unsafe {
        std::slice::from_raw_parts(buf.contents() as *const f32, size)
    };
    let nan_count = data.iter().filter(|&&v| v.is_nan()).count();
    let zero_count = data.iter().filter(|&&v| v == 0.0).count();
    println!(
        "{}: {} NaN, {} zero, first 5: {:?}",
        name,
        nan_count,
        zero_count,
        &data[..5.min(data.len())]
    );
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

            println!("=== Layer 0 Operations ===");

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let layer0 = &model.layers[0];
            let attention_norm =
                layer0.attention_norm.as_ref().expect("No norm");
            let norm_weight_buffer =
                gpu_ctx.get_or_create_buffer(attention_norm);

            // 1. RMS Norm
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

            // 2. QKV projections
            let wq = layer0.wq.as_ref().expect("No wq");
            let wk = layer0.wk.as_ref().expect("No wk");
            let wv = layer0.wv.as_ref().expect("No wv");

            let q_out_size = wq.shape[1];
            let k_out_size = wk.shape[1];
            let v_out_size = wv.shape[1];

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

            // 3. RoPE
            let head_dim = model.config.head_dim;
            let actual_n_heads = q_out_size / head_dim;
            let actual_n_kv_heads = k_out_size / head_dim;

            let q_rope_buffer = forward_pool.get(q_out_size * 4);
            let k_rope_buffer = forward_pool.get(k_out_size * 4);

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

            // 4. KV update
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

            // Memory barrier for KV cache
            use metal::ResourceRef;
            encoder.memory_barrier_with_resources(&[
                k_cache.as_ref() as &ResourceRef,
                v_cache.as_ref() as &ResourceRef,
            ]);

            // 5. Flash Attention
            let attn_out_buffer = forward_pool.get(q_out_size * 4);
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

            // 6. Output projection
            let wo = layer0.wo.as_ref().expect("No wo");
            let attn_proj_buffer = forward_pool.get(hidden_size * 4);
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

            // 7. Residual add
            let residual_buffer = forward_pool.get(hidden_size * 4);
            gpu_ctx.add_encode(
                encoder,
                &hidden_buffer,
                &attn_proj_buffer,
                &residual_buffer,
                hidden_size,
            );

            // 8. FFN norm
            let ffn_norm = layer0.ffn_norm.as_ref().expect("No ffn_norm");
            let ffn_norm_buffer = gpu_ctx.get_or_create_buffer(ffn_norm);
            let ffn_normed_buffer = forward_pool.get(hidden_size * 4);
            gpu_ctx.rms_norm_encode(
                encoder,
                &residual_buffer,
                &ffn_norm_buffer,
                &ffn_normed_buffer,
                1,
                hidden_size,
                1e-5,
            );

            // 9. FFN
            let w1 = layer0.w1.as_ref().expect("No w1");
            let w2 = layer0.w2.as_ref().expect("No w2");
            let w3 = layer0.w3.as_ref().expect("No w3");
            let intermediate_size = w1.shape[1];

            let gate_buffer = forward_pool.get(intermediate_size * 4);
            let up_buffer = forward_pool.get(intermediate_size * 4);
            let silu_buffer = forward_pool.get(intermediate_size * 4);
            let hidden_ffn_buffer = forward_pool.get(intermediate_size * 4);
            let ffn_out_buffer = forward_pool.get(hidden_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &ffn_normed_buffer,
                    w1,
                    intermediate_size,
                    gpu_ctx,
                    &gate_buffer,
                )
                .expect("W1 failed");
            model
                .gemv_encode(
                    encoder,
                    &ffn_normed_buffer,
                    w3,
                    intermediate_size,
                    gpu_ctx,
                    &up_buffer,
                )
                .expect("W3 failed");
            gpu_ctx.silu_encode(
                encoder,
                &gate_buffer,
                &silu_buffer,
                intermediate_size,
            );
            gpu_ctx.mul_encode(
                encoder,
                &silu_buffer,
                &up_buffer,
                &hidden_ffn_buffer,
                intermediate_size,
            );
            model
                .gemv_encode(
                    encoder,
                    &hidden_ffn_buffer,
                    w2,
                    hidden_size,
                    gpu_ctx,
                    &ffn_out_buffer,
                )
                .expect("W2 failed");

            // 10. Final residual
            let final_buffer = forward_pool.get(hidden_size * 4);
            gpu_ctx.add_encode(
                encoder,
                &residual_buffer,
                &ffn_out_buffer,
                &final_buffer,
                hidden_size,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Check intermediate results
            check_buffer(&normed_buffer, "After RMS norm", hidden_size);
            check_buffer(&q_buffer, "After Q projection", q_out_size);
            check_buffer(&k_buffer, "After K projection", k_out_size);
            check_buffer(&v_buffer, "After V projection", v_out_size);
            check_buffer(&q_rope_buffer, "After Q RoPE", q_out_size);
            check_buffer(&k_rope_buffer, "After K RoPE", k_out_size);
            check_buffer(&attn_out_buffer, "After Flash Attention", q_out_size);
            check_buffer(&attn_proj_buffer, "After WO projection", hidden_size);
            check_buffer(&residual_buffer, "After first residual", hidden_size);
            check_buffer(&ffn_normed_buffer, "After FFN norm", hidden_size);
            check_buffer(&gate_buffer, "After W1 (gate)", intermediate_size);
            check_buffer(&up_buffer, "After W3 (up)", intermediate_size);
            check_buffer(&silu_buffer, "After SiLU", intermediate_size);
            check_buffer(
                &hidden_ffn_buffer,
                "After gate*up",
                intermediate_size,
            );
            check_buffer(&ffn_out_buffer, "After W2", hidden_size);
            check_buffer(&final_buffer, "After final residual", hidden_size);

            // ============================================================
            // Now test Layer 1 with the same forward_pool
            // ============================================================
            println!("\n=== Layer 1 Operations (same command buffer) ===");

            // Don't clear - we want to test with the same buffers

            let layer1 = &model.layers[1];
            let attention_norm1 =
                layer1.attention_norm.as_ref().expect("No norm");
            let norm_weight_buffer1 =
                gpu_ctx.get_or_create_buffer(attention_norm1);

            // Layer 1 input is Layer 0's output
            let layer1_input = final_buffer.clone();
            check_buffer(
                &layer1_input,
                "Layer 1 input (after Layer 0 command buffer completes)",
                hidden_size,
            );

            // Create NEW command buffer and encoder - THIS WORKS
            let command_buffer2 = queue.new_command_buffer();
            let encoder2 = command_buffer2.new_compute_command_encoder();

            // 1. RMS Norm
            let normed_buffer1 = forward_pool.get_zeroed(hidden_size * 4);
            gpu_ctx.rms_norm_encode(
                encoder2,
                &layer1_input,
                &norm_weight_buffer1,
                &normed_buffer1,
                1,
                hidden_size,
                1e-5,
            );

            encoder2.end_encoding();
            command_buffer2.commit();
            command_buffer2.wait_until_completed();

            check_buffer(
                &normed_buffer1,
                "Layer 1 after RMS norm (separate command buffer)",
                hidden_size,
            );

            println!("DEBUG: About to test same command buffer");

            // ============================================================
            // Now test SAME command buffer - this should fail
            // ============================================================
            println!("\n=== Layer 0 + Layer 1 in SAME command buffer ===");

            kv_cache.clear();
            forward_pool.release_all();

            let command_buffer3 = queue.new_command_buffer();
            let encoder3 = command_buffer3.new_compute_command_encoder();

            // Layer 0 again
            let normed_buffer3 = forward_pool.get_zeroed(hidden_size * 4);
            gpu_ctx.rms_norm_encode(
                encoder3,
                &hidden_buffer,
                &norm_weight_buffer,
                &normed_buffer3,
                1,
                hidden_size,
                1e-5,
            );

            let q_buffer3 = forward_pool.get_zeroed(q_out_size * 4);
            model
                .gemv_encode(
                    encoder3,
                    &normed_buffer3,
                    wq,
                    q_out_size,
                    gpu_ctx,
                    &q_buffer3,
                )
                .expect("Q failed");
            // ... encode full Layer 0 ...
            let q_rope_buffer3 = forward_pool.get(q_out_size * 4);
            gpu_ctx.rope_neox_encode(
                encoder3,
                &q_buffer3,
                &[0],
                1,
                actual_n_heads,
                head_dim,
                10000.0,
                1.0,
                &q_rope_buffer3,
            );

            let k_buffer3 = forward_pool.get_zeroed(k_out_size * 4);
            model
                .gemv_encode(
                    encoder3,
                    &normed_buffer3,
                    wk,
                    k_out_size,
                    gpu_ctx,
                    &k_buffer3,
                )
                .expect("K failed");
            let k_rope_buffer3 = forward_pool.get(k_out_size * 4);
            gpu_ctx.rope_neox_encode(
                encoder3,
                &k_buffer3,
                &[0],
                1,
                actual_n_kv_heads,
                head_dim,
                10000.0,
                1.0,
                &k_rope_buffer3,
            );

            let v_buffer3 = forward_pool.get_zeroed(v_out_size * 4);
            model
                .gemv_encode(
                    encoder3,
                    &normed_buffer3,
                    wv,
                    v_out_size,
                    gpu_ctx,
                    &v_buffer3,
                )
                .expect("V failed");

            let k_cache0 = kv_cache.k_buffer(0);
            let v_cache0 = kv_cache.v_buffer(0);
            gpu_ctx.kv_update_k_encode(
                encoder3,
                &k_rope_buffer3,
                k_cache0,
                0,
                kv_cache.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );
            gpu_ctx.kv_update_v_encode(
                encoder3,
                &v_buffer3,
                v_cache0,
                0,
                kv_cache.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );

            encoder3.memory_barrier_with_resources(&[
                k_cache0.as_ref() as &ResourceRef,
                v_cache0.as_ref() as &ResourceRef,
            ]);

            let attn_out_buffer3 = forward_pool.get(q_out_size * 4);
            gpu_ctx.flash_attn_f16_encode(
                encoder3,
                1,
                actual_n_heads,
                actual_n_kv_heads,
                head_dim,
                1,
                kv_cache.max_seq(),
                &q_rope_buffer3,
                k_cache0,
                v_cache0,
                &attn_out_buffer3,
            );

            let attn_proj_buffer3 = forward_pool.get(hidden_size * 4);
            model
                .gemv_encode(
                    encoder3,
                    &attn_out_buffer3,
                    wo,
                    hidden_size,
                    gpu_ctx,
                    &attn_proj_buffer3,
                )
                .expect("WO failed");

            let residual_buffer3 = forward_pool.get(hidden_size * 4);
            gpu_ctx.add_encode(
                encoder3,
                &hidden_buffer,
                &attn_proj_buffer3,
                &residual_buffer3,
                hidden_size,
            );

            let ffn_normed_buffer3 = forward_pool.get(hidden_size * 4);
            gpu_ctx.rms_norm_encode(
                encoder3,
                &residual_buffer3,
                &ffn_norm_buffer,
                &ffn_normed_buffer3,
                1,
                hidden_size,
                1e-5,
            );

            let gate_buffer3 = forward_pool.get(intermediate_size * 4);
            let up_buffer3 = forward_pool.get(intermediate_size * 4);
            model
                .gemv_encode(
                    encoder3,
                    &ffn_normed_buffer3,
                    w1,
                    intermediate_size,
                    gpu_ctx,
                    &gate_buffer3,
                )
                .expect("W1 failed");
            model
                .gemv_encode(
                    encoder3,
                    &ffn_normed_buffer3,
                    w3,
                    intermediate_size,
                    gpu_ctx,
                    &up_buffer3,
                )
                .expect("W3 failed");

            let silu_buffer3 = forward_pool.get(intermediate_size * 4);
            gpu_ctx.silu_encode(
                encoder3,
                &gate_buffer3,
                &silu_buffer3,
                intermediate_size,
            );

            let hidden_ffn_buffer3 = forward_pool.get(intermediate_size * 4);
            gpu_ctx.mul_encode(
                encoder3,
                &silu_buffer3,
                &up_buffer3,
                &hidden_ffn_buffer3,
                intermediate_size,
            );

            let ffn_out_buffer3 = forward_pool.get(hidden_size * 4);
            model
                .gemv_encode(
                    encoder3,
                    &hidden_ffn_buffer3,
                    w2,
                    hidden_size,
                    gpu_ctx,
                    &ffn_out_buffer3,
                )
                .expect("W2 failed");

            let layer0_output = forward_pool.get(hidden_size * 4);
            gpu_ctx.add_encode(
                encoder3,
                &residual_buffer3,
                &ffn_out_buffer3,
                &layer0_output,
                hidden_size,
            );

            println!("Layer 0 encoded in shared buffer");

            // Now encode Layer 1 in the SAME command buffer
            let normed_buffer4 = forward_pool.get_zeroed(hidden_size * 4);
            gpu_ctx.rms_norm_encode(
                encoder3,
                &layer0_output,
                &norm_weight_buffer1,
                &normed_buffer4,
                1,
                hidden_size,
                1e-5,
            );

            println!("Layer 1 RMS norm encoded in shared buffer");

            encoder3.end_encoding();
            command_buffer3.commit();
            command_buffer3.wait_until_completed();

            println!("\nAfter execution:");
            check_buffer(&layer0_output, "Layer 0 output", hidden_size);
            check_buffer(
                &normed_buffer4,
                "Layer 1 after RMS norm",
                hidden_size,
            );
        }
    }
}
