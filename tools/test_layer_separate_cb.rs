// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test: each layer uses independent command buffer (llama.cpp approach)

use infer_train::inference::{Device, InferenceContext};
use std::cell::RefCell;
use std::rc::Rc;

fn check_buffer(name: &str, buffer: &metal::BufferRef) {
    let contents = buffer.contents() as *const f32;
    let len = buffer.length() as usize / 4;

    let num_nan =
        unsafe { (0..len).filter(|&i| (*contents.add(i)).is_nan()).count() };

    println!("{}: {} NaN", name, num_nan);
}

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
                "=== Test: Each layer uses independent command buffer ===\n"
            );
            println!(
                "Approach: llama.cpp splits graph into chunks per command buffer\n"
            );

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            // Collect command buffers for later synchronization
            let mut command_buffers = Vec::new();

            // Process layers one by one, each with its own command buffer
            let mut layer_input = hidden_buffer.clone();
            let n_layers = 2; // Test first 2 layers only

            for layer_idx in 0..n_layers {
                // Create new command buffer for this layer
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                // Encode layer
                let layer_output = model
                    .forward_layer_gpu_encode(
                        &model.layers[layer_idx],
                        layer_idx,
                        &layer_input,
                        kv_cache,
                        gpu_ctx,
                        &mut forward_pool,
                        encoder,
                    )
                    .expect(&format!("Layer {} failed", layer_idx));

                encoder.end_encoding();

                // Commit immediately (llama.cpp commits each buffer right after encoding)
                command_buffer.commit();
                command_buffers.push(command_buffer);

                println!("Layer {} encoded and committed", layer_idx);

                // Use this layer's output as next layer's input
                layer_input = layer_output;
            }

            // Wait for all command buffers to complete
            println!("\nWaiting for all command buffers to complete...");
            for (idx, cmd_buf) in command_buffers.iter().enumerate() {
                cmd_buf.wait_until_completed();
                println!("Command buffer {} completed", idx);
            }

            // Check results
            println!("\n=== Results ===\n");

            for layer_idx in 0..n_layers {
                let k_cache = kv_cache.get_k(layer_idx, 0);
                let k_nan = k_cache.iter().filter(|&&v| v.is_nan()).count();
                println!("Layer {} K cache: {} NaN", layer_idx, k_nan);
            }

            // Check final layer output
            let contents = layer_input.contents() as *const f32;
            let num_nan = unsafe {
                (0..hidden_size)
                    .filter(|&i| (*contents.add(i)).is_nan())
                    .count()
            };
            println!("\nFinal output: {} NaN", num_nan);

            if num_nan == 0 {
                println!(
                    "\n*** SUCCESS: Each layer in separate command buffer works! ***"
                );
            } else {
                println!("\n*** Still has NaN issue ***");
            }
        }
    }
}
