// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test with FRESH contexts for each run to isolate the issue

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        // Run 1: Fresh context
        println!("=== Run 1: Fresh context ===");
        let ctx1 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let embedding1 = ctx1.model.embedding_lookup_impl(1).unwrap();

        let gpu_ctx1 = ctx1.gpu_ctx.as_ref().unwrap();
        let device1 = gpu_ctx1.device();

        let hidden_buffer1 = std::rc::Rc::new(device1.new_buffer_with_data(
            embedding1.as_ptr() as *const std::ffi::c_void,
            (embedding1.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        use infer_train::inference::KVCacheGPU;
        let config1 = &ctx1.model.config;
        let mut kv_cache1 = KVCacheGPU::new(device1, config1, 2048);

        let reuse_pool1 = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut forward_pool1 = infer_train::buffer::ForwardBufferPool::new(
            device1.clone(),
            reuse_pool1.clone(),
        );

        let layer1 = &ctx1.model.layers[0];
        let queue1 = gpu_ctx1.queue();
        let command_buffer1 = queue1.new_command_buffer();
        let encoder1 = command_buffer1.new_compute_command_encoder();

        let layer_output1 = ctx1.model.forward_layer_gpu_encode(
            layer1,
            0,
            &hidden_buffer1,
            &mut kv_cache1,
            gpu_ctx1,
            &mut forward_pool1,
            encoder1,
        );

        encoder1.end_encoding();
        command_buffer1.commit();
        command_buffer1.wait_until_completed();

        if let Some(output1) = layer_output1 {
            let mut result1 = vec![0.0f32; output1.length() as usize / 4];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    output1.contents() as *const f32,
                    result1.as_mut_ptr(),
                    result1.len(),
                );
            }

            let nan_count = result1.iter().filter(|&&v| v.is_nan()).count();
            let zero_count = result1.iter().filter(|&&v| v == 0.0).count();
            println!("  Output size: {}", result1.len());
            println!("  NaN count: {}", nan_count);
            println!("  Zero count: {}", zero_count);
            println!("  First 5: {:?}", &result1[..5]);
        }

        // Drop everything
        drop(forward_pool1);
        drop(ctx1);

        // Run 2: Fresh context
        println!("\n=== Run 2: Fresh context ===");
        let ctx2 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let embedding2 = ctx2.model.embedding_lookup_impl(1).unwrap();

        let gpu_ctx2 = ctx2.gpu_ctx.as_ref().unwrap();
        let device2 = gpu_ctx2.device();

        let hidden_buffer2 = std::rc::Rc::new(device2.new_buffer_with_data(
            embedding2.as_ptr() as *const std::ffi::c_void,
            (embedding2.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let config2 = &ctx2.model.config;
        let mut kv_cache2 = KVCacheGPU::new(device2, config2, 2048);

        let reuse_pool2 = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut forward_pool2 = infer_train::buffer::ForwardBufferPool::new(
            device2.clone(),
            reuse_pool2.clone(),
        );

        let layer2 = &ctx2.model.layers[0];
        let queue2 = gpu_ctx2.queue();
        let command_buffer2 = queue2.new_command_buffer();
        let encoder2 = command_buffer2.new_compute_command_encoder();

        let layer_output2 = ctx2.model.forward_layer_gpu_encode(
            layer2,
            0,
            &hidden_buffer2,
            &mut kv_cache2,
            gpu_ctx2,
            &mut forward_pool2,
            encoder2,
        );

        encoder2.end_encoding();
        command_buffer2.commit();
        command_buffer2.wait_until_completed();

        if let Some(output2) = layer_output2 {
            let mut result2 = vec![0.0f32; output2.length() as usize / 4];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    output2.contents() as *const f32,
                    result2.as_mut_ptr(),
                    result2.len(),
                );
            }

            let nan_count = result2.iter().filter(|&&v| v.is_nan()).count();
            let zero_count = result2.iter().filter(|&&v| v == 0.0).count();
            println!("  Output size: {}", result2.len());
            println!("  NaN count: {}", nan_count);
            println!("  Zero count: {}", zero_count);
            println!("  First 5: {:?}", &result2[..5]);
        }

        drop(forward_pool2);
        drop(ctx2);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
