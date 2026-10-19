// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test: Run twice with the SAME context to see if there's state corruption

use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        // Single context
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let model = &ctx.model;
        let gpu_ctx = ctx.gpu_ctx.as_ref().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();
        let config = &model.config;
        let hidden_size = config.hidden_size;

        // Create two fresh KV caches
        let mut kv1 = KVCacheGPU::new(device, config, 2048);
        let mut kv2 = KVCacheGPU::new(device, config, 2048);

        // Get embedding
        let emb = model.embedding_lookup_impl(1).unwrap();
        println!("Embedding first 5: {:?}", &emb[..5]);

        // Upload to GPU
        let emb_buf1 = Rc::new(device.new_buffer_with_data(
            emb.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));
        let emb_buf2 = Rc::new(device.new_buffer_with_data(
            emb.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Create forward pools (SEPARATE for each run)
        let reuse_pool1 = Rc::new(RefCell::new(Vec::new()));
        let mut pool1 = infer_train::buffer::ForwardBufferPool::new(
            device.clone(),
            reuse_pool1,
        );
        let reuse_pool2 = Rc::new(RefCell::new(Vec::new()));
        let mut pool2 = infer_train::buffer::ForwardBufferPool::new(
            device.clone(),
            reuse_pool2,
        );

        println!("\n=== Run 1: Layer 0 ===");
        let cb1 = queue.new_command_buffer();
        let enc1 = cb1.new_compute_command_encoder();
        let out1 = model
            .forward_layer_gpu_encode(
                &model.layers[0],
                0,
                &emb_buf1,
                &mut kv1,
                gpu_ctx,
                &mut pool1,
                enc1,
            )
            .unwrap();
        enc1.end_encoding();
        cb1.commit();
        cb1.wait_until_completed();

        let mut h1 = vec![0.0f32; hidden_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                out1.contents() as *const f32,
                h1.as_mut_ptr(),
                hidden_size,
            );
        }
        let nan1 = h1.iter().filter(|&&v| v.is_nan()).count();
        println!("Run1 NaN count: {}", nan1);
        println!("Run1 first 5: {:?}", &h1[..5]);

        println!(
            "\n=== Run 2: Layer 0 (same context, new KV cache, new pool) ==="
        );
        let cb2 = queue.new_command_buffer();
        let enc2 = cb2.new_compute_command_encoder();
        let out2 = model
            .forward_layer_gpu_encode(
                &model.layers[0],
                0,
                &emb_buf2,
                &mut kv2,
                gpu_ctx,
                &mut pool2,
                enc2,
            )
            .unwrap();
        enc2.end_encoding();
        cb2.commit();
        cb2.wait_until_completed();

        let mut h2 = vec![0.0f32; hidden_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                out2.contents() as *const f32,
                h2.as_mut_ptr(),
                hidden_size,
            );
        }
        let nan2 = h2.iter().filter(|&&v| v.is_nan()).count();
        println!("Run2 NaN count: {}", nan2);
        println!("Run2 first 5: {:?}", &h2[..5]);

        if nan1 == 0 && nan2 > 0 {
            println!("\n✗ NaN appears in Run2 only - state corruption");
            // Find NaN positions
            for (i, (&v1, &v2)) in h1.iter().zip(h2.iter()).enumerate() {
                if v2.is_nan() {
                    println!("  Position {}: run1={:.4}, run2=NaN", i, v1);
                }
            }
        } else if nan1 > 0 || nan2 > 0 {
            println!("\n✗ NaN detected in one or both runs");
        } else {
            println!("\n✓ No NaN in either run");
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
