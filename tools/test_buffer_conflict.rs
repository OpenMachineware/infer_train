// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug buffer addresses to detect conflicts

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
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
            use metal::MTLResourceOptions;
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            use infer_train::buffer::ForwardBufferPool;
            use std::cell::RefCell;
            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let mut current_buffer = hidden_buffer.clone();
            let mut buffer_addresses: Vec<(usize, usize, *const u8)> =
                Vec::new(); // (layer, idx, ptr)

            for layer_idx in 0..5.min(model.layers.len()) {
                let layer = &model.layers[layer_idx];
                let prev_buffer_ptr = current_buffer.contents() as *const u8;

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
                    .expect("Layer forward failed");

                let new_buffer_ptr = current_buffer.contents() as *const u8;
                buffer_addresses.push((layer_idx, 0, new_buffer_ptr));
            }

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Check for duplicate buffer addresses
            println!("Buffer addresses (layer, ptr):");
            for (layer, _, ptr) in &buffer_addresses {
                println!("  Layer {}: {:?}", layer, ptr);
            }

            // Check forward_pool stats
            let stats = forward_pool.stats();
            println!("\nForward pool stats:");
            println!("  Active buffers: {}", stats.active);
            println!("  Reusable buffers: {}", stats.reusable);

            // Read output
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
            println!("\nOutput after 5 layers:");
            println!("  NaN: {}, Zero: {}", num_nan, num_zero);
            println!("  First 5: {:?}", &output[..5]);
        }
    }
}
