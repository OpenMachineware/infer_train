// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test: Does reuse pool sharing cause the corruption?

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::buffer::ForwardBufferPool;
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};

        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let config = ctx.model.config.clone();
        let gpu_ctx = ctx.gpu_ctx.as_ref().unwrap();

        // Test 1: SHARED reuse pool across runs (expected to fail)
        println!("=== Test 1: SHARED reuse pool ===");
        {
            let shared_reuse_pool =
                std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

            for run in 0..2 {
                println!("\nRun {}:", run + 1);

                let mut kv_cache =
                    KVCacheGPU::new(gpu_ctx.device(), &config, 2048);
                let mut forward_pool = ForwardBufferPool::new(
                    gpu_ctx.device().clone(),
                    shared_reuse_pool.clone(), // SHARED!
                );

                let embedding = ctx.model.embedding_lookup_impl(1).unwrap();
                let hidden_buffer =
                    std::rc::Rc::new(gpu_ctx.device().new_buffer_with_data(
                        embedding.as_ptr() as *const std::ffi::c_void,
                        (embedding.len() * 4) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    ));

                let layer = &ctx.model.layers[0];
                let queue = gpu_ctx.queue();
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                let layer_output = ctx.model.forward_layer_gpu_encode(
                    layer,
                    0,
                    &hidden_buffer,
                    &mut kv_cache,
                    gpu_ctx,
                    &mut forward_pool,
                    encoder,
                );

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                if let Some(output) = layer_output {
                    let mut result = vec![0.0f32; output.length() as usize / 4];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            output.contents() as *const f32,
                            result.as_mut_ptr(),
                            result.len(),
                        );
                    }

                    let nan_count =
                        result.iter().filter(|&&v| v.is_nan()).count();
                    let zero_count =
                        result.iter().filter(|&&v| v == 0.0).count();
                    println!(
                        "  NaN: {}, Zero: {}, First 5: {:?}",
                        nan_count,
                        zero_count,
                        &result[..5]
                    );
                }

                println!(
                    "  Reuse pool size: {}",
                    shared_reuse_pool.borrow().len()
                );
                drop(forward_pool); // Returns buffers to shared_reuse_pool
                println!(
                    "  Reuse pool size after drop: {}",
                    shared_reuse_pool.borrow().len()
                );
            }
        }

        // Test 2: FRESH reuse pool for each run (expected to work)
        println!("\n=== Test 2: FRESH reuse pool ===");
        {
            for run in 0..2 {
                println!("\nRun {}:", run + 1);

                let fresh_reuse_pool =
                    std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

                let mut kv_cache =
                    KVCacheGPU::new(gpu_ctx.device(), &config, 2048);
                let mut forward_pool = ForwardBufferPool::new(
                    gpu_ctx.device().clone(),
                    fresh_reuse_pool.clone(), // FRESH!
                );

                let embedding = ctx.model.embedding_lookup_impl(1).unwrap();
                let hidden_buffer =
                    std::rc::Rc::new(gpu_ctx.device().new_buffer_with_data(
                        embedding.as_ptr() as *const std::ffi::c_void,
                        (embedding.len() * 4) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    ));

                let layer = &ctx.model.layers[0];
                let queue = gpu_ctx.queue();
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                let layer_output = ctx.model.forward_layer_gpu_encode(
                    layer,
                    0,
                    &hidden_buffer,
                    &mut kv_cache,
                    gpu_ctx,
                    &mut forward_pool,
                    encoder,
                );

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                if let Some(output) = layer_output {
                    let mut result = vec![0.0f32; output.length() as usize / 4];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            output.contents() as *const f32,
                            result.as_mut_ptr(),
                            result.len(),
                        );
                    }

                    let nan_count =
                        result.iter().filter(|&&v| v.is_nan()).count();
                    let zero_count =
                        result.iter().filter(|&&v| v == 0.0).count();
                    println!(
                        "  NaN: {}, Zero: {}, First 5: {:?}",
                        nan_count,
                        zero_count,
                        &result[..5]
                    );
                }

                drop(forward_pool);
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
