// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test: Run 1 with old context, Run 2 with fresh context

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::buffer::ForwardBufferPool;
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};

        // Run 1: Load context, process
        println!("=== Run 1: Initial context ===");
        let ctx1 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let config = ctx1.model.config.clone();
        let gpu_ctx = ctx1.gpu_ctx.as_ref().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        let mut kv_cache1 = KVCacheGPU::new(device, &config, 2048);
        let reuse_pool1 = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

        let embedding1 = ctx1.model.embedding_lookup_impl(1).unwrap();
        let mut layer_input1 = std::rc::Rc::new(device.new_buffer_with_data(
            embedding1.as_ptr() as *const std::ffi::c_void,
            (embedding1.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        for layer_idx in 0..4 {
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool1.clone());

            let layer = &ctx1.model.layers[layer_idx];
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let layer_output = ctx1.model.forward_layer_gpu_encode(
                layer,
                layer_idx,
                &layer_input1,
                &mut kv_cache1,
                gpu_ctx,
                &mut forward_pool,
                encoder,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            if let Some(output) = layer_output {
                let mut result = vec![0.0f32; 100];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output.contents() as *const f32,
                        result.as_mut_ptr(),
                        100,
                    );
                }

                let nan_count = result.iter().filter(|v| v.is_nan()).count();
                println!("  Layer {}: NaN count = {}", layer_idx, nan_count);
                layer_input1 = output;
            }

            drop(forward_pool);
        }

        kv_cache1.advance();

        // Drop context 1
        println!("\nDropping context 1...");
        drop(ctx1);

        // Run 2: FRESH context (new model load, new Metal context)
        println!("\n=== Run 2: FRESH context (after dropping old) ===");
        let ctx2 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let gpu_ctx2 = ctx2.gpu_ctx.as_ref().unwrap();
        let device2 = gpu_ctx2.device();
        let queue2 = gpu_ctx2.queue();

        let mut kv_cache2 = KVCacheGPU::new(device2, &config, 2048);
        let reuse_pool2 = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

        let embedding2 = ctx2.model.embedding_lookup_impl(1).unwrap();
        let mut layer_input2 = std::rc::Rc::new(device2.new_buffer_with_data(
            embedding2.as_ptr() as *const std::ffi::c_void,
            (embedding2.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        for layer_idx in 0..4 {
            let mut forward_pool =
                ForwardBufferPool::new(device2.clone(), reuse_pool2.clone());

            let layer = &ctx2.model.layers[layer_idx];
            let command_buffer = queue2.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let layer_output = ctx2.model.forward_layer_gpu_encode(
                layer,
                layer_idx,
                &layer_input2,
                &mut kv_cache2,
                gpu_ctx2,
                &mut forward_pool,
                encoder,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            if let Some(output) = layer_output {
                let mut result = vec![0.0f32; 100];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output.contents() as *const f32,
                        result.as_mut_ptr(),
                        100,
                    );
                }

                let nan_count = result.iter().filter(|v| v.is_nan()).count();
                println!("  Layer {}: NaN count = {}", layer_idx, nan_count);
                layer_input2 = output;
            }

            drop(forward_pool);
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
