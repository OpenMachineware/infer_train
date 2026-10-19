// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test where NaN starts appearing

use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
use std::cell::RefCell;
use std::rc::Rc;

fn check_nan(name: &str, data: &[f32]) {
    let nan_count = data.iter().filter(|&&v| v.is_nan()).count();
    let inf_count = data.iter().filter(|&&v| v.is_infinite()).count();
    println!(
        "{}: {} NaN, {} inf, first 5: {:?}",
        name,
        nan_count,
        inf_count,
        &data[..5.min(data.len())]
    );
}

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx1 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let ctx2 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let model = &ctx1.model;
        let gpu_ctx = ctx1.gpu_ctx.as_ref().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();
        let config = &model.config;
        let hidden_size = config.hidden_size;

        // Create fresh KV caches
        let mut kv1 = KVCacheGPU::new(device, config, 2048);
        let mut kv2 = KVCacheGPU::new(device, config, 2048);

        // Get embedding
        let emb = model.embedding_lookup_impl(1).unwrap();
        check_nan("Embedding", &emb);

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

        // Process all layers
        let mut input1 = emb_buf1.clone();
        let mut input2 = emb_buf2.clone();

        println!("\n=== Processing layers ===");
        for layer_idx in 0..model.layers.len() {
            // Run 1
            let cb1 = queue.new_command_buffer();
            let enc1 = cb1.new_compute_command_encoder();
            let out1 = model
                .forward_layer_gpu_encode(
                    &model.layers[layer_idx],
                    layer_idx,
                    &input1,
                    &mut kv1,
                    gpu_ctx,
                    &mut pool1,
                    enc1,
                )
                .unwrap();
            enc1.end_encoding();
            cb1.commit();
            cb1.wait_until_completed();

            // Run 2
            let cb2 = queue.new_command_buffer();
            let enc2 = cb2.new_compute_command_encoder();
            let out2 = model
                .forward_layer_gpu_encode(
                    &model.layers[layer_idx],
                    layer_idx,
                    &input2,
                    &mut kv2,
                    gpu_ctx,
                    &mut pool2,
                    enc2,
                )
                .unwrap();
            enc2.end_encoding();
            cb2.commit();
            cb2.wait_until_completed();

            // Check for NaN after each layer
            let mut h1 = vec![0.0f32; hidden_size];
            let mut h2 = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out1.contents() as *const f32,
                    h1.as_mut_ptr(),
                    hidden_size,
                );
                std::ptr::copy_nonoverlapping(
                    out2.contents() as *const f32,
                    h2.as_mut_ptr(),
                    hidden_size,
                );
            }

            let nan1 = h1.iter().filter(|&&v| v.is_nan()).count();
            let nan2 = h2.iter().filter(|&&v| v.is_nan()).count();

            if nan1 > 0 || nan2 > 0 {
                println!(
                    "Layer {} NaN: run1={}, run2={}",
                    layer_idx, nan1, nan2
                );
                check_nan(&format!("  Run1 layer {}", layer_idx), &h1);
                check_nan(&format!("  Run2 layer {}", layer_idx), &h2);
                break;
            }

            input1 = out1;
            input2 = out2;
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
