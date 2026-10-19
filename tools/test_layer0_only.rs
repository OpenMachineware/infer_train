// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test: Just run layer 0 to narrow down non-determinism

use infer_train::inference::{Device, InferenceContext};
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx1 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let ctx2 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

        let gpu_ctx = ctx1.gpu_ctx.as_ref().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();
        let config = &ctx1.model.config;
        let hidden_size = config.hidden_size;

        // Create fresh KV caches
        let mut kv1 =
            infer_train::inference::KVCacheGPU::new(device, config, 2048);
        let mut kv2 =
            infer_train::inference::KVCacheGPU::new(device, config, 2048);

        // Get embedding
        let emb = ctx1.model.embedding_lookup_impl(1).unwrap();
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

        // Create forward pools
        let reuse_pool = Rc::new(RefCell::new(Vec::new()));
        let mut pool1 = infer_train::buffer::ForwardBufferPool::new(
            device.clone(),
            reuse_pool.clone(),
        );
        let reuse_pool2 = Rc::new(RefCell::new(Vec::new()));
        let mut pool2 = infer_train::buffer::ForwardBufferPool::new(
            device.clone(),
            reuse_pool2,
        );

        // Run layer 0 twice
        println!("\n=== Run 1 ===");
        let cb1 = queue.new_command_buffer();
        let enc1 = cb1.new_compute_command_encoder();
        let out1 = ctx1
            .model
            .forward_layer_gpu_encode(
                &ctx1.model.layers[0],
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

        // Read output
        let mut h1 = vec![0.0f32; hidden_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                out1.contents() as *const f32,
                h1.as_mut_ptr(),
                hidden_size,
            );
        }
        println!("Layer 0 output first 5: {:?}", &h1[..5]);

        println!("\n=== Run 2 ===");
        let cb2 = queue.new_command_buffer();
        let enc2 = cb2.new_compute_command_encoder();
        let out2 = ctx2
            .model
            .forward_layer_gpu_encode(
                &ctx2.model.layers[0],
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

        // Read output
        let mut h2 = vec![0.0f32; hidden_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                out2.contents() as *const f32,
                h2.as_mut_ptr(),
                hidden_size,
            );
        }
        println!("Layer 0 output first 5: {:?}", &h2[..5]);

        // Compare
        let max_diff: f32 = h1
            .iter()
            .zip(h2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        println!("\nMax diff: {:.6}", max_diff);
        if max_diff < 0.001 {
            println!("✓ Deterministic");
        } else {
            println!("✗ NON-DETERMINISTIC");
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
