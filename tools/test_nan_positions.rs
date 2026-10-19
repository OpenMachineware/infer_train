// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug: Find exactly where NaN values are in the output buffer

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
        let hidden_size = config.hidden_size;

        println!("Hidden size: {}", hidden_size);

        for run in 0..2 {
            println!("\n=== Run {} ===", run + 1);

            let mut kv_cache = KVCacheGPU::new(device, &config, 2048);
            let reuse_pool =
                std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

            let embedding = ctx.model.embedding_lookup_impl(1).unwrap();
            let hidden_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                embedding.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            let layer = &ctx.model.layers[0];
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

                // Find all NaN positions
                let nan_positions: Vec<usize> = result
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| v.is_nan())
                    .map(|(i, _)| i)
                    .collect();

                println!("Total values: {}", result.len());
                println!("NaN count: {}", nan_positions.len());

                if !nan_positions.is_empty() {
                    println!(
                        "First 20 NaN positions: {:?}",
                        &nan_positions[..20.min(nan_positions.len())]
                    );

                    // Check what the values at those positions were in Run 1 (if this is Run 2)
                    // Also check values around the NaN positions
                    for &pos in nan_positions.iter().take(5) {
                        let start = pos.saturating_sub(2);
                        let end = (pos + 3).min(result.len());
                        println!(
                            "  Around position {}: {:?}",
                            pos,
                            &result[start..end]
                        );
                    }
                }
            }

            drop(forward_pool);
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
