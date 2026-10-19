// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test complete forward pass step by step

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::buffer::ForwardBufferPool;
        use infer_train::inference::{Device, InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::cell::RefCell;
        use std::rc::Rc;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();
        let config = ctx.model.config.clone();
        let hidden_size = config.hidden_size;

        let mut kv_cache = KVCacheGPU::new(device, &config, 2048);

        let reuse_pool: Rc<RefCell<Vec<Rc<metal::Buffer>>>> =
            Rc::new(RefCell::new(Vec::new()));
        let mut forward_pool =
            ForwardBufferPool::new(device.clone(), reuse_pool.clone());

        // 1. Embedding
        let hidden = ctx.model.embedding_lookup_impl(1).unwrap();
        println!("=== Embedding ===");
        println!("First 5: {:?}", &hidden[..5]);

        let hidden_buffer = Rc::new(device.new_buffer_with_data(
            hidden.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // 2. Process layers
        println!("\n=== Processing Layers ===");
        let mut layer_input = hidden_buffer.clone();
        let n_layers = ctx.model.layers.len();

        for layer_idx in 0..n_layers {
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let layer_output = ctx
                .model
                .forward_layer_gpu_encode(
                    &ctx.model.layers[layer_idx],
                    layer_idx,
                    &layer_input,
                    &mut kv_cache,
                    &gpu_ctx,
                    &mut forward_pool,
                    encoder,
                )
                .unwrap();

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            layer_input = layer_output;

            if layer_idx == n_layers - 1 {
                // Check last layer output
                let mut h = vec![0.0f32; 100];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        layer_input.contents() as *const f32,
                        h.as_mut_ptr(),
                        100,
                    );
                }
                let nan_count = h.iter().filter(|v| v.is_nan()).count();
                println!(
                    "Layer {} (last) output: {} NaN, first 5 = {:?}",
                    layer_idx,
                    nan_count,
                    &h[..5]
                );
            }
        }

        kv_cache.advance();

        // 3. Final norm
        println!("\n=== Final Norm ===");
        let final_norm = ctx.model.norm.as_ref().unwrap();
        let final_norm_buffer = gpu_ctx.get_or_create_buffer(final_norm);
        let final_hidden_buffer = forward_pool.get(hidden_size * 4);

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        gpu_ctx.rms_norm_encode(
            encoder,
            &layer_input,
            &final_norm_buffer,
            &final_hidden_buffer,
            1,
            hidden_size,
            1e-5,
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Check norm output
        let mut normed = vec![0.0f32; 100];
        unsafe {
            std::ptr::copy_nonoverlapping(
                final_hidden_buffer.contents() as *const f32,
                normed.as_mut_ptr(),
                100,
            );
        }
        let nan_count = normed.iter().filter(|v| v.is_nan()).count();
        println!(
            "After final norm: {} NaN, first 5 = {:?}",
            nan_count,
            &normed[..5]
        );

        // 4. Output projection (CPU fallback for transposed)
        println!("\n=== Output Projection ===");
        let output = ctx.model.output.as_ref().unwrap();
        println!("Output weight dtype: {:?}", output.dtype);
        println!("Output weight shape: {:?}", output.shape);

        // Read hidden to CPU
        let mut hidden_cpu = vec![0.0f32; hidden_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                final_hidden_buffer.contents() as *const f32,
                hidden_cpu.as_mut_ptr(),
                hidden_size,
            );
        }

        let vocab_size =
            if output.shape.len() == 2 && output.shape[1] == hidden_size {
                output.shape[0]
            } else {
                output.shape[0]
            };

        let result =
            if output.shape.len() == 2 && output.shape[1] == hidden_size {
                ctx.model.gemv_transposed(&hidden_cpu, output, vocab_size)
            } else {
                ctx.model.gemv(&hidden_cpu, output, vocab_size)
            };

        match result {
            Some(logits) => {
                let nan_count = logits.iter().filter(|v| v.is_nan()).count();
                println!(
                    "Output: {} logits, {} NaN, first 5 = {:?}",
                    logits.len(),
                    nan_count,
                    &logits[..5.min(logits.len())]
                );
            }
            None => println!("Output projection failed"),
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
