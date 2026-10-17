// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug what Layer 1 reads as input

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
    println!("Model loaded successfully");

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

            // Test: Same command buffer, but read Layer 0 output BEFORE encoding Layer 1
            println!(
                "\n=== Test: Read Layer 0 output before encoding Layer 1 ==="
            );

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
            println!(
                "Layer 0 encoded, output buffer: {:p}",
                layer0_out.as_ref()
            );

            // Now we need to encode a "copy" operation to save Layer 0's output
            // But since we haven't committed yet, the GPU hasn't run!
            // Let's add a simple add operation to copy the buffer
            let copy_buffer = forward_pool.get(hidden_size * 4);
            gpu_ctx.add_encode(
                encoder,
                &layer0_out,
                &forward_pool.get_zeroed(hidden_size * 4), // add zero = copy
                &copy_buffer,
                hidden_size,
            );
            println!("Copy operation encoded");

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

            // Read the copied version of Layer 0 output
            let mut copy_data = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    copy_buffer.contents() as *const f32,
                    copy_data.as_mut_ptr(),
                    hidden_size,
                );
            }
            let copy_nan = copy_data.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "\nCopied Layer 0 output: {} NaN, first 5: {:?}",
                copy_nan,
                &copy_data[..5.min(copy_data.len())]
            );

            // Read Layer 0 output directly
            let mut layer0_data = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    layer0_out.contents() as *const f32,
                    layer0_data.as_mut_ptr(),
                    hidden_size,
                );
            }
            let layer0_nan =
                layer0_data.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "Direct Layer 0 output: {} NaN, first 5: {:?}",
                layer0_nan,
                &layer0_data[..5.min(layer0_data.len())]
            );

            // Check K caches
            let k_cache0 = kv_cache.get_k(0, 0);
            let k_cache1 = kv_cache.get_k(1, 0);
            println!(
                "\nK cache 0: first 5: {:?}",
                &k_cache0[..5.min(k_cache0.len())]
            );
            println!(
                "K cache 1: first 5: {:?}",
                &k_cache1[..5.min(k_cache1.len())]
            );
        }
    }
}
