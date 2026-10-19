// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test if fresh MetalContext fixes the issue (simplified)

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::buffer::ForwardBufferPool;
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};

        // Load full context once
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let config = ctx.model.config.clone();

        // Run with SAME MetalContext (expected to fail on Run 2)
        println!("=== Test: SAME MetalContext ===");
        {
            let gpu_ctx = ctx.gpu_ctx.as_ref().unwrap();

            for run in 0..2 {
                println!("\nRun {}:", run + 1);

                let mut kv_cache =
                    KVCacheGPU::new(gpu_ctx.device(), &config, 2048);
                let reuse_pool =
                    std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
                let mut forward_pool = ForwardBufferPool::new(
                    gpu_ctx.device().clone(),
                    reuse_pool.clone(),
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

        // Run with FRESH MetalContext (using same model)
        println!("\n=== Test: FRESH MetalContext (same model) ===");
        {
            for run in 0..2 {
                println!("\nRun {}:", run + 1);

                // Create FRESH MetalContext for each run (but use same model)
                let fresh_gpu_ctx =
                    infer_train::quant::vec_dot::gpu::metal::MetalContext::new(
                    )
                    .unwrap();

                let mut kv_cache =
                    KVCacheGPU::new(fresh_gpu_ctx.device(), &config, 2048);
                let reuse_pool =
                    std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
                let mut forward_pool = ForwardBufferPool::new(
                    fresh_gpu_ctx.device().clone(),
                    reuse_pool.clone(),
                );

                let embedding = ctx.model.embedding_lookup_impl(1).unwrap();
                let hidden_buffer = std::rc::Rc::new(
                    fresh_gpu_ctx.device().new_buffer_with_data(
                        embedding.as_ptr() as *const std::ffi::c_void,
                        (embedding.len() * 4) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    ),
                );

                let layer = &ctx.model.layers[0];
                let queue = fresh_gpu_ctx.queue();
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                let layer_output = ctx.model.forward_layer_gpu_encode(
                    layer,
                    0,
                    &hidden_buffer,
                    &mut kv_cache,
                    &fresh_gpu_ctx,
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
                drop(fresh_gpu_ctx);
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
