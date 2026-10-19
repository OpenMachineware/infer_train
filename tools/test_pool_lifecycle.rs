// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test: Keep forward_pool alive until next layer is done

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
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        println!("Test: Keep forward_pool alive until next layer starts");

        for run in 0..2 {
            println!("\n=== Run {} ===", run + 1);

            let mut kv_cache = KVCacheGPU::new(device, &config, 2048);
            let reuse_pool =
                std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

            let embedding = ctx.model.embedding_lookup_impl(1).unwrap();
            let mut layer_input =
                std::rc::Rc::new(device.new_buffer_with_data(
                    embedding.as_ptr() as *const std::ffi::c_void,
                    (embedding.len() * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                ));

            let mut prev_pool: Option<ForwardBufferPool> = None;

            for layer_idx in 0..4 {
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

                if let Some(output) = layer_output {
                    let mut result = vec![0.0f32; 100];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            output.contents() as *const f32,
                            result.as_mut_ptr(),
                            100,
                        );
                    }

                    let nan_count =
                        result.iter().filter(|v| v.is_nan()).count();
                    println!(
                        "  Layer {}: NaN count = {}",
                        layer_idx, nan_count
                    );

                    layer_input = output;
                }

                // Drop previous pool (so layer N-1's output is returned to reuse_pool)
                // But layer N is already using a new pool with fresh buffers
                if let Some(pool) = prev_pool.take() {
                    drop(pool);
                }

                // Keep this pool for next iteration
                prev_pool = Some(forward_pool);
            }

            // Drop last pool
            if let Some(pool) = prev_pool {
                drop(pool);
            }

            kv_cache.advance();
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
