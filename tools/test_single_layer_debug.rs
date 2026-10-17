// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug single layer GPU forward

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (ctx.gpu_ctx.as_ref(), ctx.gpu_kv_cache.as_mut())
        {
            let model = &ctx.model;
            let hidden_size = model.config.hidden_size;
            let n_layers = model.config.n_layers;
            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            // Get embedding for token 9707
            let hidden =
                model.embedding_lookup_impl(9707).expect("No embedding");
            println!(
                "Embedding: first 5: {:?}",
                &hidden[..5.min(hidden.len())]
            );

            // Upload to GPU
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // Create shared reuse pool
            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            // Test each layer individually
            for layer_idx in 0..n_layers.min(3) {
                // Clear KV cache
                kv_cache.clear();
                kv_cache.reset_seq_len();

                // Create fresh pool for this layer
                let mut forward_pool =
                    ForwardBufferPool::new(device.clone(), reuse_pool.clone());

                // Create command buffer
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                // Run layer
                let layer = &model.layers[layer_idx];
                let output_buffer = model
                    .forward_layer_gpu_encode(
                        layer,
                        layer_idx,
                        &hidden_buffer,
                        kv_cache,
                        gpu_ctx,
                        &mut forward_pool,
                        encoder,
                    )
                    .expect(&format!("Layer {} failed", layer_idx));

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                // Read output
                let mut output = vec![0.0f32; hidden_size];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        output_buffer.contents() as *const f32,
                        output.as_mut_ptr(),
                        hidden_size,
                    );
                }

                let num_nan = output.iter().filter(|&&v| v.is_nan()).count();
                let num_zero = output.iter().filter(|&&v| v == 0.0).count();
                println!(
                    "Layer {} output: {} NaN, {} zero, first 5: {:?}",
                    layer_idx,
                    num_nan,
                    num_zero,
                    &output[..5.min(output.len())]
                );

                // Check K cache
                let k_cache = kv_cache.get_k(layer_idx, 0);
                let k_nan = k_cache.iter().filter(|&&v| v.is_nan()).count();
                let k_zero = k_cache.iter().filter(|&&v| v == 0.0).count();
                println!(
                    "  K cache: {} NaN, {} zero, first 5: {:?}",
                    k_nan,
                    k_zero,
                    &k_cache[..5.min(k_cache.len())]
                );

                // Drop pool to release buffers
                drop(forward_pool);
            }
        }
    }
}
