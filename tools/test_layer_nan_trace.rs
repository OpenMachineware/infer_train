// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Trace NaN propagation through layers

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use metal::MTLResourceOptions;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();

        println!("=== Creating InferenceContext ===");
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();
        let config = ctx.model.config.clone();
        let hidden_size = config.hidden_size;

        // === Test 1: First prefill ===
        println!("\n=== Test 1: First Prefill ===");
        {
            let mut kv = KVCacheGPU::new(device, &config, 2048);

            // Check embedding
            let embedding = ctx.model.embedding_lookup_impl(1).unwrap();
            let embed_nan = embedding.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "Embedding: {} NaN, first 5: {:?}",
                embed_nan,
                &embedding[..5.min(embedding.len())]
            );

            // Upload to GPU and check
            let hidden_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                embedding.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            // Read back from GPU
            let readback: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    hidden_buffer.contents() as *const f32,
                    hidden_size,
                )
                .to_vec()
            };
            let gpu_nan = readback.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "GPU buffer: {} NaN, first 5: {:?}",
                gpu_nan,
                &readback[..5.min(readback.len())]
            );

            // Run first layer
            let queue = gpu_ctx.queue();
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let layer = &ctx.model.layers[0];
            let attention_norm = layer.attention_norm.as_ref().unwrap();
            let norm_weight_buffer =
                gpu_ctx.get_or_create_buffer(attention_norm);

            // Create output buffer
            let normed_buffer = std::rc::Rc::new(device.new_buffer(
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            gpu_ctx.rms_norm_encode(
                encoder,
                &hidden_buffer,
                &norm_weight_buffer,
                &normed_buffer,
                1,
                hidden_size,
                1e-5,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Check RMS norm output
            let normed_readback: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    normed_buffer.contents() as *const f32,
                    hidden_size,
                )
                .to_vec()
            };
            let norm_nan =
                normed_readback.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "After RMS norm: {} NaN, first 5: {:?}",
                norm_nan,
                &normed_readback[..5.min(normed_readback.len())]
            );
        }

        // === Test 2: Second prefill ===
        println!("\n=== Test 2: Second Prefill ===");
        {
            let mut kv = KVCacheGPU::new(device, &config, 2048);

            // Check embedding
            let embedding = ctx.model.embedding_lookup_impl(1).unwrap();
            let embed_nan = embedding.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "Embedding: {} NaN, first 5: {:?}",
                embed_nan,
                &embedding[..5.min(embedding.len())]
            );

            // Upload to GPU and check
            let hidden_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                embedding.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            // Read back from GPU
            let readback: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    hidden_buffer.contents() as *const f32,
                    hidden_size,
                )
                .to_vec()
            };
            let gpu_nan = readback.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "GPU buffer: {} NaN, first 5: {:?}",
                gpu_nan,
                &readback[..5.min(readback.len())]
            );

            // Run first layer
            let queue = gpu_ctx.queue();
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let layer = &ctx.model.layers[0];
            let attention_norm = layer.attention_norm.as_ref().unwrap();
            let norm_weight_buffer =
                gpu_ctx.get_or_create_buffer(attention_norm);

            // Create output buffer
            let normed_buffer = std::rc::Rc::new(device.new_buffer(
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            gpu_ctx.rms_norm_encode(
                encoder,
                &hidden_buffer,
                &norm_weight_buffer,
                &normed_buffer,
                1,
                hidden_size,
                1e-5,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Check RMS norm output
            let normed_readback: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    normed_buffer.contents() as *const f32,
                    hidden_size,
                )
                .to_vec()
            };
            let norm_nan =
                normed_readback.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "After RMS norm: {} NaN, first 5: {:?}",
                norm_nan,
                &normed_readback[..5.min(normed_readback.len())]
            );
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
