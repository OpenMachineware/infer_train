// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test forward_layer_gpu_encode chaining

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

            // Test 1: Run layer 0 and layer 1 in SEPARATE command buffers
            println!("\n=== Test 1: Separate command buffers ===");

            // Layer 0
            let mut forward_pool0 =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer0 = queue.new_command_buffer();
            let encoder0 = command_buffer0.new_compute_command_encoder();

            let layer0_out = model
                .forward_layer_gpu_encode(
                    &model.layers[0],
                    0,
                    &hidden_buffer,
                    kv_cache,
                    gpu_ctx,
                    &mut forward_pool0,
                    encoder0,
                )
                .expect("Layer 0 failed");

            encoder0.end_encoding();
            command_buffer0.commit();
            command_buffer0.wait_until_completed();

            let k_cache0_sep = kv_cache.get_k(0, 0);
            println!(
                "Layer 0 K cache: first 5: {:?}",
                &k_cache0_sep[..5.min(k_cache0_sep.len())]
            );

            // Layer 1 (using layer 0 output as input)
            let mut forward_pool1 =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer1 = queue.new_command_buffer();
            let encoder1 = command_buffer1.new_compute_command_encoder();

            let layer1_out = model
                .forward_layer_gpu_encode(
                    &model.layers[1],
                    1,
                    &layer0_out,
                    kv_cache,
                    gpu_ctx,
                    &mut forward_pool1,
                    encoder1,
                )
                .expect("Layer 1 failed");

            encoder1.end_encoding();
            command_buffer1.commit();
            command_buffer1.wait_until_completed();

            let k_cache1_sep = kv_cache.get_k(1, 0);
            println!(
                "Layer 1 K cache: first 5: {:?}",
                &k_cache1_sep[..5.min(k_cache1_sep.len())]
            );

            // Reset KV cache for test 2
            kv_cache.clear();

            // Test 2: Run layer 0 and layer 1 in SAME command buffer
            println!("\n=== Test 2: Same command buffer ===");

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Layer 0
            let layer0_out = model
                .forward_layer_gpu_encode(
                    &model.layers[0],
                    0,
                    &hidden_buffer,
                    kv_cache,
                    gpu_ctx,
                    &mut forward_pool,
                    encoder,
                )
                .expect("Layer 0 failed");
            println!("Layer 0 encoded");

            // Layer 1
            let layer1_out = model
                .forward_layer_gpu_encode(
                    &model.layers[1],
                    1,
                    &layer0_out,
                    kv_cache,
                    gpu_ctx,
                    &mut forward_pool,
                    encoder,
                )
                .expect("Layer 1 failed");
            println!("Layer 1 encoded");

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let k_cache0_same = kv_cache.get_k(0, 0);
            let k_cache1_same = kv_cache.get_k(1, 0);
            println!(
                "Layer 0 K cache: first 5: {:?}",
                &k_cache0_same[..5.min(k_cache0_same.len())]
            );
            println!(
                "Layer 1 K cache: first 5: {:?}",
                &k_cache1_same[..5.min(k_cache1_same.len())]
            );

            // Check layer 1 output
            let mut output = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    layer1_out.contents() as *const f32,
                    output.as_mut_ptr(),
                    hidden_size,
                );
            }
            let num_nan = output.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "Final output: {} NaN, first 5: {:?}",
                num_nan,
                &output[..5.min(output.len())]
            );
        }
    }
}
