// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test to check if buffers are being reused incorrectly

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

            // Create shared reuse pool
            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            println!("=== Test: Check buffer pointer reuse ===\n");

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Layer 0: Get all buffer pointers
            println!("Layer 0 buffers:");
            let normed0 = forward_pool.get_zeroed(hidden_size * 4);
            println!("  normed0: {:p}", &*normed0 as *const _);

            let q0 = forward_pool.get_zeroed(hidden_size * 4);
            println!("  q0: {:p}", &*q0 as *const _);

            let k0 = forward_pool.get_zeroed(hidden_size * 4);
            println!("  k0: {:p}", &*k0 as *const _);

            let v0 = forward_pool.get_zeroed(hidden_size * 4);
            println!("  v0: {:p}", &*v0 as *const _);

            let layer0_out = forward_pool.get_zeroed(hidden_size * 4);
            println!("  layer0_out: {:p}", &*layer0_out as *const _);

            // Layer 1: Get all buffer pointers
            println!("\nLayer 1 buffers:");
            let normed1 = forward_pool.get_zeroed(hidden_size * 4);
            println!("  normed1: {:p}", &*normed1 as *const _);

            // Check if normed1 reuses layer0_out!
            if Rc::ptr_eq(&normed1, &layer0_out) {
                println!("  *** ALIASING DETECTED: normed1 == layer0_out ***");
            } else {
                println!("  normed1 != layer0_out (OK)");
            }

            let q1 = forward_pool.get_zeroed(hidden_size * 4);
            println!("  q1: {:p}", &*q1 as *const _);

            let k1 = forward_pool.get_zeroed(hidden_size * 4);
            println!("  k1: {:p}", &*k1 as *const _);

            let v1 = forward_pool.get_zeroed(hidden_size * 4);
            println!("  v1: {:p}", &*v1 as *const _);

            let layer1_out = forward_pool.get_zeroed(hidden_size * 4);
            println!("  layer1_out: {:p}", &*layer1_out as *const _);

            println!("\nBuffer stats: {:?}", forward_pool.stats());
        }
    }
}
