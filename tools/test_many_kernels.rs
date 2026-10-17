// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test if many kernels in one command buffer causes issues

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
        if let (Some(ref gpu_ctx), Some(ref _kv_cache)) =
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
                "=== Test: 40 simple operations in one command buffer ===\n"
            );

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let input = forward_pool.register(hidden_buffer);

            // Do 40 operations: alternating add and mul
            let mut current = input.clone();
            for i in 0..40 {
                let output = forward_pool.get_zeroed(hidden_size * 4);

                if i % 2 == 0 {
                    // Add: output = input + input
                    gpu_ctx.add_encode(
                        &encoder,
                        &current,
                        &current,
                        &output,
                        hidden_size,
                    );
                } else {
                    // Mul: output = input * input
                    gpu_ctx.mul_encode(
                        &encoder,
                        &current,
                        &current,
                        &output,
                        hidden_size,
                    );
                }

                current = output;
                println!("Operation {} encoded", i);
            }

            let final_buffer = current;

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            println!("\n=== Results ===\n");
            check_buffer("final_buffer", &final_buffer);

            // Check if any intermediate buffer has NaN
            println!("\nChecking some intermediate buffers:");
            let all_buffers = forward_pool.stats();
            println!("Total buffers: {}", all_buffers.active);
        }
    }
}
