// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test if GEMV causes issues when run multiple times

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

            let layer0 = &model.layers[0];
            let wq = layer0.wq.as_ref().unwrap();
            let q_out_size = wq.shape[1];

            println!(
                "=== Test: GEMV (Q projection) twice in one command buffer ===\n"
            );

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let input = forward_pool.register(hidden_buffer);

            // GEMV 1
            let q1 = forward_pool.get_zeroed(q_out_size * 4);
            model
                .gemv_encode(
                    encoder,
                    &input,
                    hidden_size,
                    wq,
                    q_out_size,
                    gpu_ctx,
                    &q1,
                )
                .unwrap();
            println!("GEMV 1 encoded");

            // GEMV 2 (using same input and weight)
            let q2 = forward_pool.get_zeroed(q_out_size * 4);
            model
                .gemv_encode(
                    encoder,
                    &input,
                    hidden_size,
                    wq,
                    q_out_size,
                    gpu_ctx,
                    &q2,
                )
                .unwrap();
            println!("GEMV 2 encoded");

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            println!("\n=== Results ===\n");
            check_buffer("q1", &q1);
            check_buffer("q2", &q2);

            // Check if outputs match (should be identical)
            let contents1 = q1.contents() as *const f32;
            let contents2 = q2.contents() as *const f32;
            let mut diff_count = 0;
            let mut max_diff = 0.0f32;
            unsafe {
                for i in 0..q_out_size {
                    let v1 = *contents1.add(i);
                    let v2 = *contents2.add(i);
                    let diff = (v1 - v2).abs();
                    if diff > 1e-6 {
                        diff_count += 1;
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                }
            }

            if diff_count > 0 {
                println!(
                    "\n*** Outputs differ! {} elements, max diff = {} ***",
                    diff_count, max_diff
                );
            } else {
                println!("\n*** Outputs match! ***");
            }
        }
    }
}
