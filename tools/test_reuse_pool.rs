// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test if buffer reuse causes the second run to produce zeros

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

        // Get embedding for token 1
        let embedding = ctx.model.embedding_lookup_impl(1).unwrap();
        println!("Embedding first 5: {:?}", &embedding[..5]);

        // Upload to GPU
        let gpu_ctx = ctx.gpu_ctx.as_ref().unwrap();
        let device = gpu_ctx.device();

        let hidden_buffer = std::rc::Rc::new(device.new_buffer_with_data(
            embedding.as_ptr() as *const std::ffi::c_void,
            (embedding.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Run layer 0 twice - WITH and WITHOUT reuse pool
        for test_case in 0..2 {
            let use_reuse_pool = test_case == 0;
            println!(
                "\n=== Test: {} reuse pool ===",
                if use_reuse_pool { "WITH" } else { "WITHOUT" }
            );

            // Create an EMPTY reuse pool for each test case
            let reuse_pool = if use_reuse_pool {
                std::rc::Rc::new(std::cell::RefCell::new(Vec::new()))
            } else {
                // For "without reuse pool", use a pool that won't return buffers
                std::rc::Rc::new(std::cell::RefCell::new(Vec::new()))
            };

            for run in 0..2 {
                println!("\n  Run {}:", run + 1);

                // Create fresh KV cache for each run
                use infer_train::inference::KVCacheGPU;
                let config = &ctx.model.config;
                let mut kv_cache = KVCacheGPU::new(device, config, 2048);

                // Create fresh forward pool
                let mut forward_pool =
                    infer_train::buffer::ForwardBufferPool::new(
                        device.clone(),
                        reuse_pool.clone(),
                    );

                // Get layer 0 weights
                let layer = &ctx.model.layers[0];

                // Run layer 0 forward
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
                    // Read the output
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
                    println!("    Output size: {}", result.len());
                    println!("    NaN count: {}", nan_count);
                    println!("    Zero count: {}", zero_count);
                    println!("    First 5: {:?}", &result[..5]);

                    // Check pool stats
                    let stats = forward_pool.stats();
                    println!(
                        "    Pool stats: active={}, reusable={}",
                        stats.active, stats.reusable
                    );
                } else {
                    println!("    Layer forward returned None");
                }

                // Drop the forward pool - buffers go to reuse_pool
                drop(forward_pool);

                // Check reuse pool size
                println!(
                    "    Reuse pool size after drop: {}",
                    reuse_pool.borrow().len()
                );
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
