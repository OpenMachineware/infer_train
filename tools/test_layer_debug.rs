// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

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
            println!("Number of layers: {}", model.layers.len());

            // Test layers 0 and 1 (like working test)
            for test_layer_idx in 0..2.min(model.layers.len()) {
                println!("\n=== Testing Layer {} ===", test_layer_idx);

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

                let mut forward_pool =
                    ForwardBufferPool::new(device.clone(), reuse_pool.clone());

                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                let layer_output = model
                    .forward_layer_gpu_encode(
                        &model.layers[test_layer_idx],
                        test_layer_idx,
                        &hidden_buffer,
                        kv_cache,
                        gpu_ctx,
                        &mut forward_pool,
                        encoder,
                    )
                    .expect(&format!("Layer {} failed", test_layer_idx));

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                // Check result
                let contents = layer_output.contents() as *const f32;
                let num_nan = unsafe {
                    (0..hidden_size)
                        .filter(|&i| (*contents.add(i)).is_nan())
                        .count()
                };

                let k_cache = kv_cache.get_k(test_layer_idx, 0);
                let k_nan = k_cache.iter().filter(|&&v| v.is_nan()).count();

                println!("  Output: {} NaN / {}", num_nan, hidden_size);
                println!("  K cache: {} NaN", k_nan);

                if num_nan == 0 && k_nan == 0 {
                    println!("  ✓ Layer {} OK", test_layer_idx);
                } else {
                    println!("  ✗ Layer {} HAS NaN!", test_layer_idx);
                }
            }
        }
    }
}
