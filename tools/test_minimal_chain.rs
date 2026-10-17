// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test minimal chaining: just RMS norm across multiple "layers"

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
            let n_layers = model.config.n_layers;
            let n_heads = model.config.n_heads;
            let n_kv_heads = model.config.n_kv_heads;
            let head_dim = model.config.head_dim;
            println!(
                "Config: hidden={}, layers={}, heads={}, kv_heads={}, head_dim={}",
                hidden_size, n_layers, n_heads, n_kv_heads, head_dim
            );

            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            // Get embedding
            let hidden =
                model.embedding_lookup_impl(9707).expect("No embedding");
            println!(
                "Embedding: first 5: {:?}",
                &hidden[..5.min(hidden.len())]
            );

            let input_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // Create shared reuse pool
            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            // Create ONE command buffer
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Create ONE forward pool
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            // Test chaining: apply RMS norm 3 times (simulating 3 layers)
            let layer0_norm = &model.layers[0].attention_norm;
            let layer1_norm = &model.layers[1].attention_norm;
            let layer2_norm = &model.layers[2].attention_norm;

            if layer0_norm.is_none()
                || layer1_norm.is_none()
                || layer2_norm.is_none()
            {
                println!("No attention norm weights");
                return;
            }

            let norm0_buffer =
                gpu_ctx.get_or_create_buffer(layer0_norm.as_ref().unwrap());
            let norm1_buffer =
                gpu_ctx.get_or_create_buffer(layer1_norm.as_ref().unwrap());
            let norm2_buffer =
                gpu_ctx.get_or_create_buffer(layer2_norm.as_ref().unwrap());

            // "Layer 0" RMS norm + QKV projection + RoPE
            let normed0 = forward_pool.get(hidden_size * 4);
            gpu_ctx.rms_norm_encode(
                encoder,
                &input_buffer,
                &norm0_buffer,
                &normed0,
                1,
                hidden_size,
                1e-5,
            );

            let wq0 = model.layers[0].wq.as_ref().unwrap();
            let wk0 = model.layers[0].wk.as_ref().unwrap();
            let wv0 = model.layers[0].wv.as_ref().unwrap();
            let q0_size = wq0.shape[1];
            let k0_size = wk0.shape[1];
            let v0_size = wv0.shape[1];

            let q0 = forward_pool.get(q0_size * 4);
            let k0 = forward_pool.get(k0_size * 4);
            let v0 = forward_pool.get(v0_size * 4);

            model.gemv_encode(encoder, &normed0, wq0, q0_size, gpu_ctx, &q0);
            model.gemv_encode(encoder, &normed0, wk0, k0_size, gpu_ctx, &k0);
            model.gemv_encode(encoder, &normed0, wv0, v0_size, gpu_ctx, &v0);

            let q_rope0 = forward_pool.get(q0_size * 4);
            let k_rope0 = forward_pool.get(k0_size * 4);

            let head_dim = model.config.head_dim;
            let n_heads = q0_size / head_dim;
            let n_kv_heads = k0_size / head_dim;

            gpu_ctx.rope_neox_encode(
                encoder,
                &q0,
                &[0],
                1,
                n_heads,
                head_dim,
                10000.0,
                1.0,
                &q_rope0,
            );
            gpu_ctx.rope_neox_encode(
                encoder,
                &k0,
                &[0],
                1,
                n_kv_heads,
                head_dim,
                10000.0,
                1.0,
                &k_rope0,
            );

            // KV update for layer 0
            let k_cache0 = forward_pool.register(kv_cache.k_buffer(0).clone());
            let v_cache0 = forward_pool.register(kv_cache.v_buffer(0).clone());
            println!(
                "k_rope0 length: {}, k_cache0 length: {}",
                k_rope0.length(),
                k_cache0.length()
            );
            println!(
                "Dispatching KV update for layer 0: pos=0, max_seq={}, n_kv_heads={}, head_dim={}",
                kv_cache.max_seq(),
                n_kv_heads,
                head_dim
            );
            gpu_ctx.kv_update_k_encode(
                encoder,
                &k_rope0,
                &*k_cache0,
                0, // pos
                kv_cache.max_seq(),
                n_kv_heads,
                head_dim,
            );
            gpu_ctx.kv_update_v_encode(
                encoder,
                &v0,
                &*v_cache0,
                0, // pos
                kv_cache.max_seq(),
                n_kv_heads,
                head_dim,
            );

            // Memory barrier to ensure KV cache writes are visible
            use metal::ResourceRef;
            encoder.memory_barrier_with_resources(&[
                k_cache0.as_ref() as &ResourceRef,
                v_cache0.as_ref() as &ResourceRef,
            ]);
            println!("Encoded 'layer 0' (norm + QKV + RoPE + KV update)");

            // Skip layer 1 for now - just test layer 0
            /*
            // "Layer 1" RMS norm + QKV + RoPE
            let normed1 = forward_pool.get(hidden_size * 4);
            gpu_ctx.rms_norm_encode(
                encoder,
                &q_rope0,  // Use previous output as input
                &norm1_buffer,
                &normed1,
                1,
                hidden_size,
                1e-5,
            );

            let wq1 = model.layers[1].wq.as_ref().unwrap();
            let wk1 = model.layers[1].wk.as_ref().unwrap();
            let wv1 = model.layers[1].wv.as_ref().unwrap();
            let q1_size = wq1.shape[1];
            let k1_size = wk1.shape[1];
            let v1_size = wv1.shape[1];

            let q1 = forward_pool.get(q1_size * 4);
            let k1 = forward_pool.get(k1_size * 4);
            let v1 = forward_pool.get(v1_size * 4);

            model.gemv_encode(encoder, &normed1, wq1, q1_size, gpu_ctx, &q1);
            model.gemv_encode(encoder, &normed1, wk1, k1_size, gpu_ctx, &k1);
            model.gemv_encode(encoder, &normed1, wv1, v1_size, gpu_ctx, &v1);

            let q_rope1 = forward_pool.get(q1_size * 4);
            let k_rope1 = forward_pool.get(k1_size * 4);

            gpu_ctx.rope_neox_encode(
                encoder,
                &q1,
                &[0],
                1,
                q1_size / head_dim,
                head_dim,
                10000.0,
                1.0,
                &q_rope1,
            );
            gpu_ctx.rope_neox_encode(
                encoder,
                &k1,
                &[0],
                1,
                k1_size / head_dim,
                head_dim,
                10000.0,
                1.0,
                &k_rope1,
            );

            // KV update for layer 1
            let k_cache1 = forward_pool.register(kv_cache.k_buffer(1).clone());
            let v_cache1 = forward_pool.register(kv_cache.v_buffer(1).clone());
            gpu_ctx.kv_update_k_encode(
                encoder,
                &k_rope1,
                &*k_cache1,
                0,  // pos
                kv_cache.max_seq(),
                k1_size / head_dim,
                head_dim,
            );
            gpu_ctx.kv_update_v_encode(
                encoder,
                &v1,
                &*v_cache1,
                0,  // pos
                kv_cache.max_seq(),
                k1_size / head_dim,
                head_dim,
            );

            // Memory barrier to ensure KV cache writes are visible
            encoder.memory_barrier_with_resources(&[
                k_cache1.as_ref() as &ResourceRef,
                v_cache1.as_ref() as &ResourceRef,
            ]);
            println!("Encoded 'layer 1' (norm + QKV + RoPE + KV update)");
            */

            // Final output buffers
            let out0 = k_rope0;
            let out1 = normed0;
            let out2 = k0;

            // Also read K cache
            let k_cache0_vals = kv_cache.get_k(0, 0);
            let k_cache1_vals = kv_cache.get_k(1, 0);

            // Read raw bytes from K cache buffer
            let k_cache0_buf = kv_cache.k_buffer(0);
            let raw_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    k_cache0_buf.contents() as *const u8,
                    32.min(k_cache0_buf.length() as usize),
                )
            };
            println!("K cache 0 raw bytes (first 32): {:?}", raw_bytes);

            // Commit and wait
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read all outputs
            let mut result0 = vec![0.0f32; hidden_size];
            let mut result1 = vec![0.0f32; hidden_size];
            let mut result2 = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out0.contents() as *const f32,
                    result0.as_mut_ptr(),
                    hidden_size,
                );
                std::ptr::copy_nonoverlapping(
                    out1.contents() as *const f32,
                    result1.as_mut_ptr(),
                    hidden_size,
                );
                std::ptr::copy_nonoverlapping(
                    out2.contents() as *const f32,
                    result2.as_mut_ptr(),
                    hidden_size,
                );
            }

            println!(
                "Output 0: {} NaN, {} zero, first 5: {:?}",
                result0.iter().filter(|&&v| v.is_nan()).count(),
                result0.iter().filter(|&&v| v == 0.0).count(),
                &result0[..5.min(result0.len())]
            );
            println!(
                "Output 1: {} NaN, {} zero, first 5: {:?}",
                result1.iter().filter(|&&v| v.is_nan()).count(),
                result1.iter().filter(|&&v| v == 0.0).count(),
                &result1[..5.min(result1.len())]
            );
            println!(
                "Output 2: {} NaN, {} zero, first 5: {:?}",
                result2.iter().filter(|&&v| v.is_nan()).count(),
                result2.iter().filter(|&&v| v == 0.0).count(),
                &result2[..5.min(result2.len())]
            );

            println!(
                "K cache 0: {} NaN, {} zero, first 5: {:?}",
                k_cache0_vals.iter().filter(|&&v| v.is_nan()).count(),
                k_cache0_vals.iter().filter(|&&v| v == 0.0).count(),
                &k_cache0_vals[..5.min(k_cache0_vals.len())]
            );
            println!(
                "K cache 1: {} NaN, {} zero, first 5: {:?}",
                k_cache1_vals.iter().filter(|&&v| v.is_nan()).count(),
                k_cache1_vals.iter().filter(|&&v| v == 0.0).count(),
                &k_cache1_vals[..5.min(k_cache1_vals.len())]
            );
        }
    }
}
