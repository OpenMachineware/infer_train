// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test if Flash Attention causes issues when run twice

use infer_train::inference::{Device, InferenceContext};
use std::cell::RefCell;
use std::rc::Rc;

fn check_buffer(name: &str, buffer: &metal::BufferRef) {
    let contents = buffer.contents() as *const f32;
    let len = buffer.length() as usize / 4;

    let num_nan =
        unsafe { (0..len).filter(|&i| (*contents.add(i)).is_nan()).count() };

    println!("{}: {} NaN", name, num_nan);
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

            let layer0 = &model.layers[0];
            let wq = layer0.wq.as_ref().unwrap();
            let wk = layer0.wk.as_ref().unwrap();
            let q_out_size = wq.shape[1];
            let k_out_size = wk.shape[1];
            let actual_n_heads = q_out_size / head_dim;
            let actual_n_kv_heads = k_out_size / head_dim;

            println!(
                "=== Test: Flash Attention twice in one command buffer ===\n"
            );

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let hidden_buffer = forward_pool.register(hidden_buffer);

            // Simple test: Run flash attention with dummy inputs
            let pos = 0;

            // Layer 0 K/V cache
            let k_cache0 = kv_cache.k_buffer(0);
            let v_cache0 = kv_cache.v_buffer(0);

            // Layer 1 K/V cache
            let k_cache1 = kv_cache.k_buffer(1);
            let v_cache1 = kv_cache.v_buffer(1);

            // Create some dummy Q, K, V
            let q0 = forward_pool.get_zeroed(q_out_size * 4);
            let k0 = forward_pool.get_zeroed(k_out_size * 4);
            let v0 = forward_pool.get_zeroed(k_out_size * 4);

            // Initialize with 1.0 values
            unsafe {
                let q_ptr = q0.contents() as *mut f32;
                let k_ptr = k0.contents() as *mut f32;
                let v_ptr = v0.contents() as *mut f32;
                for i in 0..q_out_size {
                    *q_ptr.add(i) = 1.0;
                }
                for i in 0..k_out_size {
                    *k_ptr.add(i) = 1.0;
                    *v_ptr.add(i) = 1.0;
                }
            }

            // Update K cache for layer 0
            gpu_ctx.kv_update_k_encode(
                encoder,
                &k0,
                k_cache0,
                pos,
                kv_cache.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );
            gpu_ctx.kv_update_v_encode(
                encoder,
                &v0,
                v_cache0,
                pos,
                kv_cache.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );

            use metal::ResourceRef;
            encoder.memory_barrier_with_resources(&[
                k_cache0.as_ref() as &ResourceRef,
                v_cache0.as_ref() as &ResourceRef,
            ]);

            // Flash attention 1
            let attn_out0 = forward_pool.get_zeroed(q_out_size * 4);
            gpu_ctx.flash_attn_f16_encode(
                encoder,
                1,
                actual_n_heads,
                actual_n_kv_heads,
                head_dim,
                pos + 1,
                kv_cache.max_seq(),
                &q0,
                k_cache0,
                v_cache0,
                &attn_out0,
            );

            println!("Flash attention 0 encoded");

            // Update K cache for layer 1
            gpu_ctx.kv_update_k_encode(
                encoder,
                &k0,
                k_cache1,
                pos,
                kv_cache.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );
            gpu_ctx.kv_update_v_encode(
                encoder,
                &v0,
                v_cache1,
                pos,
                kv_cache.max_seq(),
                actual_n_kv_heads,
                head_dim,
            );

            encoder.memory_barrier_with_resources(&[
                k_cache1.as_ref() as &ResourceRef,
                v_cache1.as_ref() as &ResourceRef,
            ]);

            // Flash attention 2
            let attn_out1 = forward_pool.get_zeroed(q_out_size * 4);
            gpu_ctx.flash_attn_f16_encode(
                encoder,
                1,
                actual_n_heads,
                actual_n_kv_heads,
                head_dim,
                pos + 1,
                kv_cache.max_seq(),
                &q0,
                k_cache1,
                v_cache1,
                &attn_out1,
            );

            println!("Flash attention 1 encoded");

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            println!("\n=== Results ===\n");
            check_buffer("attn_out0", &attn_out0);
            check_buffer("attn_out1", &attn_out1);

            let k0_cache = kv_cache.get_k(0, 0);
            let k1_cache = kv_cache.get_k(1, 0);
            let k0_nan = k0_cache.iter().filter(|&&v| v.is_nan()).count();
            let k1_nan = k1_cache.iter().filter(|&&v| v.is_nan()).count();

            println!("K cache layer 0: {} NaN", k0_nan);
            println!("K cache layer 1: {} NaN", k1_nan);

            if k1_nan > 0 || k0_nan > 0 {
                println!("\n*** Flash Attention has issues! ***");
            } else {
                println!("\n*** Flash Attention works fine! ***");
            }
        }
    }
}
