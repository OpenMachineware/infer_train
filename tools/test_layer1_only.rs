// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Layer 1 specifically on Run 1 and Run 2

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::buffer::ForwardBufferPool;
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};

        // Run 1
        println!("=== Run 1: Process Layer 0 then Layer 1 ===");
        let ctx1 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let config = ctx1.model.config.clone();
        let gpu_ctx1 = ctx1.gpu_ctx.as_ref().unwrap();
        let device = gpu_ctx1.device();
        let queue = gpu_ctx1.queue();

        let mut kv_cache1 = KVCacheGPU::new(device, &config, 2048);
        let reuse_pool1 = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

        let embedding1 = ctx1.model.embedding_lookup_impl(1).unwrap();
        let emb_buffer1 = std::rc::Rc::new(device.new_buffer_with_data(
            embedding1.as_ptr() as *const std::ffi::c_void,
            (embedding1.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Layer 0
        let mut forward_pool1 =
            ForwardBufferPool::new(device.clone(), reuse_pool1.clone());
        let layer0 = &ctx1.model.layers[0];
        let cb1 = queue.new_command_buffer();
        let enc1 = cb1.new_compute_command_encoder();

        let layer0_output1 = ctx1
            .model
            .forward_layer_gpu_encode(
                layer0,
                0,
                &emb_buffer1,
                &mut kv_cache1,
                gpu_ctx1,
                &mut forward_pool1,
                enc1,
            )
            .unwrap();

        enc1.end_encoding();
        cb1.commit();
        cb1.wait_until_completed();
        drop(forward_pool1);

        let mut h0_1 = vec![0.0f32; 100];
        unsafe {
            std::ptr::copy_nonoverlapping(
                layer0_output1.contents() as *const f32,
                h0_1.as_mut_ptr(),
                100,
            );
        }
        println!("Run 1 Layer 0 output first 5: {:?}", &h0_1[..5]);

        // Layer 1
        let mut forward_pool1 =
            ForwardBufferPool::new(device.clone(), reuse_pool1.clone());
        let layer1 = &ctx1.model.layers[1];
        let cb1 = queue.new_command_buffer();
        let enc1 = cb1.new_compute_command_encoder();

        let layer1_output1 = ctx1
            .model
            .forward_layer_gpu_encode(
                layer1,
                1,
                &layer0_output1,
                &mut kv_cache1,
                gpu_ctx1,
                &mut forward_pool1,
                enc1,
            )
            .unwrap();

        enc1.end_encoding();
        cb1.commit();
        cb1.wait_until_completed();
        drop(forward_pool1);

        let mut h1_1 = vec![0.0f32; 100];
        unsafe {
            std::ptr::copy_nonoverlapping(
                layer1_output1.contents() as *const f32,
                h1_1.as_mut_ptr(),
                100,
            );
        }
        println!("Run 1 Layer 1 output first 5: {:?}", &h1_1[..5]);
        let nan1 = h1_1.iter().filter(|v| v.is_nan()).count();
        println!("Run 1 Layer 1 NaN count: {}", nan1);

        // Drop context 1
        drop(ctx1);

        // Run 2
        println!("\n=== Run 2: Process Layer 0 then Layer 1 ===");
        let ctx2 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let gpu_ctx2 = ctx2.gpu_ctx.as_ref().unwrap();
        let device2 = gpu_ctx2.device();
        let queue2 = gpu_ctx2.queue();

        let mut kv_cache2 = KVCacheGPU::new(device2, &config, 2048);
        let reuse_pool2 = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

        let embedding2 = ctx2.model.embedding_lookup_impl(1).unwrap();
        let emb_buffer2 = std::rc::Rc::new(device2.new_buffer_with_data(
            embedding2.as_ptr() as *const std::ffi::c_void,
            (embedding2.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Layer 0
        let mut forward_pool2 =
            ForwardBufferPool::new(device2.clone(), reuse_pool2.clone());
        let layer0 = &ctx2.model.layers[0];
        let cb2 = queue2.new_command_buffer();
        let enc2 = cb2.new_compute_command_encoder();

        let layer0_output2 = ctx2
            .model
            .forward_layer_gpu_encode(
                layer0,
                0,
                &emb_buffer2,
                &mut kv_cache2,
                gpu_ctx2,
                &mut forward_pool2,
                enc2,
            )
            .unwrap();

        enc2.end_encoding();
        cb2.commit();
        cb2.wait_until_completed();
        drop(forward_pool2);

        let mut h0_2 = vec![0.0f32; 100];
        unsafe {
            std::ptr::copy_nonoverlapping(
                layer0_output2.contents() as *const f32,
                h0_2.as_mut_ptr(),
                100,
            );
        }
        println!("Run 2 Layer 0 output first 5: {:?}", &h0_2[..5]);

        // Layer 1
        let mut forward_pool2 =
            ForwardBufferPool::new(device2.clone(), reuse_pool2.clone());
        let layer1 = &ctx2.model.layers[1];
        let cb2 = queue2.new_command_buffer();
        let enc2 = cb2.new_compute_command_encoder();

        let layer1_output2 = ctx2
            .model
            .forward_layer_gpu_encode(
                layer1,
                1,
                &layer0_output2,
                &mut kv_cache2,
                gpu_ctx2,
                &mut forward_pool2,
                enc2,
            )
            .unwrap();

        enc2.end_encoding();
        cb2.commit();
        cb2.wait_until_completed();

        let mut h1_2 = vec![0.0f32; 100];
        unsafe {
            std::ptr::copy_nonoverlapping(
                layer1_output2.contents() as *const f32,
                h1_2.as_mut_ptr(),
                100,
            );
        }
        println!("Run 2 Layer 1 output first 5: {:?}", &h1_2[..5]);
        let nan2 = h1_2.iter().filter(|v| v.is_nan()).count();
        println!("Run 2 Layer 1 NaN count: {}", nan2);

        println!("\n=== Summary ===");
        println!("Run 1 Layer 0 → Layer 1: {} NaN", nan1);
        println!("Run 2 Layer 0 → Layer 1: {} NaN", nan2);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
