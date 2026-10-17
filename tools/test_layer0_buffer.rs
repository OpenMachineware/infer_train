// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test if Layer 1 encoding overwrites Layer 0 output buffer

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

            println!(
                "Test: Encode both layers, check Layer 0 output BEFORE commit"
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

            println!("Layer 0 encoded");
            println!(
                "Layer 0 output buffer address: {:p}",
                layer0_out.as_ref()
            );
            println!("Active buffers: {}", forward_pool.stats().active);

            // Check if layer0_out is in the active buffer list
            let is_registered = forward_pool.stats().active > 0;
            println!("layer0_out is registered: {}", is_registered);

            // Layer 1 - this might overwrite layer0_out if it's reused
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
            println!("Active buffers: {}", forward_pool.stats().active);
            println!(
                "Layer 1 output buffer address: {:p}",
                layer1_out.as_ref()
            );

            // Check if layer0_out and layer1_out point to the same buffer
            let same_buffer = Rc::ptr_eq(&layer0_out, &layer1_out);
            println!("layer0_out == layer1_out: {}", same_buffer);

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read results
            let mut layer0_data = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    layer0_out.contents() as *const f32,
                    layer0_data.as_mut_ptr(),
                    hidden_size,
                );
            }

            let nan_count = layer0_data.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "\nLayer 0 output: {} NaN, first 5: {:?}",
                nan_count,
                &layer0_data[..5.min(layer0_data.len())]
            );

            let nan_positions: Vec<usize> = layer0_data
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_nan())
                .map(|(i, _)| i)
                .collect();
            if !nan_positions.is_empty() {
                println!(
                    "NaN positions: {:?}",
                    &nan_positions[..10.min(nan_positions.len())]
                );
            }
        }
    }
}
