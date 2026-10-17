// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test sequential layer encoding in shared command buffer

use infer_train::inference::{Device, InferenceContext};
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
            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            // Get embedding
            let hidden =
                model.embedding_lookup_impl(9707).expect("No embedding");
            println!(
                "Embedding: first 5: {:?}",
                &hidden[..5.min(hidden.len())]
            );

            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // Create shared reuse pool
            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            // Clear KV cache
            kv_cache.clear();
            kv_cache.reset_seq_len();

            // Create ONE command buffer for all layers
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Create ONE forward pool for all layers
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            // Process layers sequentially in SINGLE command buffer
            let mut current_buffer = hidden_buffer.clone();
            for (layer_idx, layer) in model.layers.iter().enumerate().take(2) {
                println!("[DEBUG] Processing layer {}", layer_idx);
                current_buffer = model
                    .forward_layer_gpu_encode(
                        layer,
                        layer_idx,
                        &current_buffer,
                        kv_cache,
                        gpu_ctx,
                        &mut forward_pool,
                        encoder,
                    )
                    .expect(&format!("Layer {} failed", layer_idx));
            }

            // Commit and wait
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read final output
            let mut output = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    current_buffer.contents() as *const f32,
                    output.as_mut_ptr(),
                    hidden_size,
                );
            }

            let num_nan = output.iter().filter(|&&v| v.is_nan()).count();
            let num_zero = output.iter().filter(|&&v| v == 0.0).count();
            println!(
                "Final output: {} NaN, {} zero, first 5: {:?}",
                num_nan,
                num_zero,
                &output[..5.min(output.len())]
            );

            // Check K cache for layer 0 and layer 1
            let k_cache0 = kv_cache.get_k(0, 0);
            println!(
                "K cache layer 0: first 5: {:?}",
                &k_cache0[..5.min(k_cache0.len())]
            );
            let k_cache1 = kv_cache.get_k(1, 0);
            println!(
                "K cache layer 1: first 5: {:?}",
                &k_cache1[..5.min(k_cache1.len())]
            );
        }
    }
}
