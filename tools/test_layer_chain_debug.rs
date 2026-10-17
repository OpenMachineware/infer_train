// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug layer chaining by checking intermediate values after GPU execution

use infer_train::inference::{Device, InferenceContext};
use std::cell::RefCell;
use std::rc::Rc;

fn check_buffer(name: &str, buffer: &metal::BufferRef) {
    let contents = buffer.contents() as *const f32;
    let len = buffer.length() as usize / 4;
    let slice = unsafe { std::slice::from_raw_parts(contents, len.min(10)) };

    let num_nan =
        unsafe { (0..len).filter(|&i| (*contents.add(i)).is_nan()).count() };

    println!(
        "{}: {} NaN (len={}), first 5: {:?}",
        name,
        num_nan,
        len,
        &slice[..5.min(slice.len())]
    );
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
                "=== Debug: Same command buffer, check all intermediate buffers ===\n"
            );

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Layer 0
            println!("Encoding Layer 0...");
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

            // Layer 1
            println!("Encoding Layer 1...");
            let _layer1_out = model
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

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            println!("\nAfter GPU execution:");
            println!("--- Layer 0 ---");
            check_buffer("layer0_out", &layer0_out);

            println!("\n--- Layer 1 input ---");
            println!("(layer0_out is Layer 1's input, same as above)");

            // Read K caches
            let k0 = kv_cache.get_k(0, 0);
            let k1 = kv_cache.get_k(1, 0);
            let k0_nan = k0.iter().filter(|&&v| v.is_nan()).count();
            let k1_nan = k1.iter().filter(|&&v| v.is_nan()).count();

            println!("\n--- K Cache ---");
            println!(
                "Layer 0 K cache: {} NaN, first 5: {:?}",
                k0_nan,
                &k0[..5.min(k0.len())]
            );
            println!(
                "Layer 1 K cache: {} NaN, first 5: {:?}",
                k1_nan,
                &k1[..5.min(k1.len())]
            );

            // Check if Layer 1 K cache is corrupted
            if k1_nan > 0 {
                println!("\n*** Layer 1 K cache is corrupted! ***");

                // Try to understand where the NaN came from
                // Check if it's a specific pattern
                let k1_first_20 = &k1[..20.min(k1.len())];
                println!("First 20 values: {:?}", k1_first_20);
            }
        }
    }
}
