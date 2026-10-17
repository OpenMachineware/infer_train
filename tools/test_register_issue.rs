// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test if register() causes issues

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

            let hidden =
                model.embedding_lookup_impl(9707).expect("No embedding");
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            // Test 1: WITHOUT register() - just pass layer0_out directly
            println!("Test 1: Layer 1 uses layer0_out directly (no register)");
            kv_cache.clear();
            {
                let mut forward_pool =
                    ForwardBufferPool::new(device.clone(), reuse_pool.clone());
                forward_pool.release_all(); // Clear reuse pool
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

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

                // Create Layer 1 input WITHOUT calling register
                // Just pass layer0_out reference
                let layer1_input = layer0_out.clone();

                let layer1_out = model
                    .forward_layer_gpu_encode(
                        &model.layers[1],
                        1,
                        &layer1_input, // This will call register internally
                        kv_cache,
                        gpu_ctx,
                        &mut forward_pool,
                        encoder,
                    )
                    .expect("Layer 1 failed");

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                let mut data = vec![0.0f32; hidden_size];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        layer0_out.contents() as *const f32,
                        data.as_mut_ptr(),
                        hidden_size,
                    );
                }
                let nan_count = data.iter().filter(|&&v| v.is_nan()).count();
                println!("Layer 0 output: {} NaN", nan_count);
            }
        }
    }
}
