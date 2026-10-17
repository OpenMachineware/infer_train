// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test separate encoding but reading together

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
                "=== Test: Encode in separate command buffers, read together ===\n"
            );

            // Layer 0 in first command buffer
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
            println!("Layer 0 encoded");

            // Layer 1 in second command buffer
            let mut forward_pool1 =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer1 = queue.new_command_buffer();
            let encoder1 = command_buffer1.new_compute_command_encoder();

            let _layer1_out = model
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
            println!("Layer 1 encoded");

            // Commit BOTH before reading
            println!("\nCommitting both command buffers...");
            command_buffer0.commit();
            command_buffer1.commit();

            println!("Waiting for completion...");
            command_buffer0.wait_until_completed();
            command_buffer1.wait_until_completed();

            println!("\n=== Results ===\n");
            check_buffer("layer0_out", &layer0_out);

            let k0 = kv_cache.get_k(0, 0);
            let k1 = kv_cache.get_k(1, 0);
            let k0_nan = k0.iter().filter(|&&v| v.is_nan()).count();
            let k1_nan = k1.iter().filter(|&&v| v.is_nan()).count();

            println!("K cache layer 0: {} NaN", k0_nan);
            println!("K cache layer 1: {} NaN", k1_nan);

            if k1_nan > 0 || k0_nan > 0 {
                println!(
                    "\n*** Issue persists even with separate buffers! ***"
                );
            } else {
                println!("\n*** Separate buffers work! ***");
            }
        }
    }
}
