// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug Layer 1 K cache update

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

            // Run Layer 0 first
            println!("=== Step 1: Run Layer 0 ===");
            let layer0_out = {
                let mut forward_pool =
                    ForwardBufferPool::new(device.clone(), reuse_pool.clone());
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                let output = model
                    .forward_layer_gpu_encode(
                        &model.layers[0],
                        0,
                        &hidden_buffer,
                        kv_cache,
                        gpu_ctx,
                        &mut forward_pool,
                        encoder,
                    )
                    .expect("Layer 0 failed");

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                let k0 = kv_cache.get_k(0, 0);
                println!(
                    "Layer 0 K cache: {} NaN, first 5: {:?}",
                    k0.iter().filter(|&&v| v.is_nan()).count(),
                    &k0[..5.min(k0.len())]
                );

                // Check Layer 0 output
                let mut layer0_data = vec![0.0f32; hidden_size];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output.contents() as *const f32,
                        layer0_data.as_mut_ptr(),
                        hidden_size,
                    );
                }
                println!(
                    "Layer 0 output: {} NaN, first 5: {:?}",
                    layer0_data.iter().filter(|&&v| v.is_nan()).count(),
                    &layer0_data[..5.min(layer0_data.len())]
                );

                output
            };

            // Now manually encode Layer 1 step by step
            println!("\n=== Step 2: Encode Layer 1 step by step ===");

            // Get Layer 0 output as input
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            // CRITICAL: Register layer0_out to keep it alive during GPU execution
            let layer1_input = forward_pool.register(layer0_out);
            println!("Registered layer0_out in forward_pool");

            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let layer1 = &model.layers[1];
            let head_dim = model.config.head_dim;

            // Attention norm - use layer1_input (registered layer0_out)
            let norm_weight = gpu_ctx
                .get_or_create_buffer(layer1.attention_norm.as_ref().unwrap());
            let normed = forward_pool.get_zeroed(hidden_size * 4);
            gpu_ctx.rms_norm_encode(
                encoder,
                &layer1_input,
                &norm_weight,
                &normed,
                1,
                hidden_size,
                1e-5,
            );
            println!("Encoded Layer 1 RMS norm (input from Layer 0 output)");

            // QKV projections
            let wq = layer1.wq.as_ref().unwrap();
            let wk = layer1.wk.as_ref().unwrap();
            let wv = layer1.wv.as_ref().unwrap();

            let q_out_size = wq.shape[1];
            let k_out_size = wk.shape[1];
            let v_out_size = wv.shape[1];
            let actual_n_heads = q_out_size / head_dim;
            let actual_n_kv_heads = k_out_size / head_dim;

            let q = forward_pool.get_zeroed(q_out_size * 4);
            let k = forward_pool.get_zeroed(k_out_size * 4);
            let v = forward_pool.get_zeroed(v_out_size * 4);

            model
                .gemv_encode(
                    encoder, &normed, 1024, wq, q_out_size, gpu_ctx, &q,
                )
                .expect("Q failed");
            model
                .gemv_encode(
                    encoder, &normed, 1024, wk, k_out_size, gpu_ctx, &k,
                )
                .expect("K failed");
            model
                .gemv_encode(
                    encoder, &normed, 1024, wv, v_out_size, gpu_ctx, &v,
                )
                .expect("V failed");
            println!("Encoded Layer 1 QKV projections");

            // RoPE
            let q_rope = forward_pool.get_zeroed(q_out_size * 4);
            let k_rope = forward_pool.get_zeroed(k_out_size * 4);
            gpu_ctx.rope_neox_encode(
                encoder,
                &q,
                &[0],
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
                &[0],
                1,
                actual_n_kv_heads,
                head_dim,
                10000.0,
                1.0,
                &k_rope,
            );
            println!("Encoded Layer 1 RoPE");

            // KV update
            let k_cache1 = kv_cache.k_buffer(1);
            let v_cache1 = kv_cache.v_buffer(1);

            gpu_ctx.kv_update_k_encode(
                encoder,
                &k_rope,
                k_cache1,
                0,
                kv_cache.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );
            gpu_ctx.kv_update_v_encode(
                encoder,
                &v,
                v_cache1,
                0,
                kv_cache.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );
            println!("Encoded Layer 1 KV update");

            // Memory barrier
            use metal::ResourceRef;
            encoder.memory_barrier_with_resources(&[
                k_cache1.as_ref() as &ResourceRef,
                v_cache1.as_ref() as &ResourceRef,
            ]);

            // Commit and wait for GPU execution
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            println!("GPU execution completed");

            // NOW we can read buffers (after GPU finishes)
            println!("\nAfter GPU execution:");
            let mut normed_data = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    normed.contents() as *const f32,
                    normed_data.as_mut_ptr(),
                    hidden_size,
                );
            }
            let normed_nan =
                normed_data.iter().filter(|&&v| v.is_nan()).count();
            let normed_zero = normed_data.iter().filter(|&&v| v == 0.0).count();
            println!(
                "  normed: {} NaN, {} zero, first 5: {:?}",
                normed_nan,
                normed_zero,
                &normed_data[..5.min(normed_data.len())]
            );

            let mut q_data = vec![0.0f32; q_out_size];
            let mut k_data = vec![0.0f32; k_out_size];
            let mut v_data = vec![0.0f32; v_out_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    q.contents() as *const f32,
                    q_data.as_mut_ptr(),
                    q_out_size,
                );
                std::ptr::copy_nonoverlapping(
                    k.contents() as *const f32,
                    k_data.as_mut_ptr(),
                    k_out_size,
                );
                std::ptr::copy_nonoverlapping(
                    v.contents() as *const f32,
                    v_data.as_mut_ptr(),
                    v_out_size,
                );
            }
            println!(
                "  Q: {} NaN, {} zero, first 5: {:?}",
                q_data.iter().filter(|&&v| v.is_nan()).count(),
                q_data.iter().filter(|&&v| v == 0.0).count(),
                &q_data[..5.min(q_data.len())]
            );
            println!(
                "  K: {} NaN, {} zero, first 5: {:?}",
                k_data.iter().filter(|&&v| v.is_nan()).count(),
                k_data.iter().filter(|&&v| v == 0.0).count(),
                &k_data[..5.min(k_data.len())]
            );
            println!(
                "  V: {} NaN, {} zero, first 5: {:?}",
                v_data.iter().filter(|&&v| v.is_nan()).count(),
                v_data.iter().filter(|&&v| v == 0.0).count(),
                &v_data[..5.min(v_data.len())]
            );

            let k_after = kv_cache.get_k(1, 0);
            println!(
                "  K cache 1: {} NaN, first 5: {:?}",
                k_after.iter().filter(|&&v| v.is_nan()).count(),
                &k_after[..5.min(k_after.len())]
            );

            let mut k_rope_data = vec![0.0f32; k_out_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    k_rope.contents() as *const f32,
                    k_rope_data.as_mut_ptr(),
                    k_out_size,
                );
            }
            let k_rope_nan =
                k_rope_data.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "  k_rope buffer: {} NaN, first 5: {:?}",
                k_rope_nan,
                &k_rope_data[..5.min(k_rope_data.len())]
            );
        }
    }
}
