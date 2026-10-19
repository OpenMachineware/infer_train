// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test multi-layer forward to find where corruption happens

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::buffer::ForwardBufferPool;
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use metal::ResourceRef;

        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let config = ctx.model.config.clone();
        let gpu_ctx = ctx.gpu_ctx.as_ref().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();
        let n_layers = ctx.model.layers.len();

        println!("Model has {} layers", n_layers);

        // Test: Run through layers one by one, check output after each
        println!("\n=== Test: Layer-by-layer output check ===");

        for run in 0..2 {
            println!("\n--- Run {} ---", run + 1);

            let mut kv_cache = KVCacheGPU::new(device, &config, 2048);
            let reuse_pool =
                std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

            // Get embedding
            let embedding = ctx.model.embedding_lookup_impl(1).unwrap();
            let mut layer_input =
                std::rc::Rc::new(device.new_buffer_with_data(
                    embedding.as_ptr() as *const std::ffi::c_void,
                    (embedding.len() * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                ));

            // Run through each layer
            for layer_idx in 0..n_layers {
                let mut forward_pool =
                    ForwardBufferPool::new(device.clone(), reuse_pool.clone());

                let layer = &ctx.model.layers[layer_idx];
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                let layer_output = ctx.model.forward_layer_gpu_encode(
                    layer,
                    layer_idx,
                    &layer_input,
                    &mut kv_cache,
                    gpu_ctx,
                    &mut forward_pool,
                    encoder,
                );

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                // Check output
                if let Some(ref output) = layer_output {
                    let mut result = vec![0.0f32; 1024]; // Just check first 1024
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            output.contents() as *const f32,
                            result.as_mut_ptr(),
                            1024,
                        );
                    }

                    let nan_count =
                        result.iter().filter(|&&v| v.is_nan()).count();
                    let zero_count =
                        result.iter().filter(|&&v| v == 0.0).count();

                    if nan_count > 0 || run == 0 && layer_idx == 0 {
                        println!(
                            "  Layer {}: NaN={}, Zero={}, First 5: {:?}",
                            layer_idx,
                            nan_count,
                            zero_count,
                            &result[..5]
                        );
                    }

                    if nan_count > 0 {
                        println!(
                            "  ❌ CORRUPTION DETECTED at layer {}",
                            layer_idx
                        );
                        break;
                    }
                }

                // Update for next layer
                layer_input = layer_output.unwrap();

                // Drop pool to return buffers
                drop(forward_pool);
            }

            kv_cache.advance();
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
