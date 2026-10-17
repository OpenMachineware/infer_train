// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test layer by layer to find where NaN starts

use infer_train::inference::{Device, InferenceContext};
use std::cell::RefCell;
use std::rc::Rc;

fn check_buffer(buf: &metal::Buffer, name: &str, size: usize) -> usize {
    let data = unsafe {
        std::slice::from_raw_parts(buf.contents() as *const f32, size)
    };
    let nan_count = data.iter().filter(|&&v| v.is_nan()).count();
    if nan_count > 0 {
        let positions: Vec<usize> = data
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_nan())
            .map(|(i, _)| i)
            .collect();
        println!(
            "{}: {} NaN at {:?}",
            name,
            nan_count,
            &positions[..5.min(positions.len())]
        );
    } else {
        println!("{}: 0 NaN ✓", name);
    }
    nan_count
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

            let n_layers = model.layers.len();
            println!("Model has {} layers\n", n_layers);

            // Test progressive layer accumulation
            for max_layer in 0..n_layers.min(4) {
                println!("=== Test: Encode layers 0..={} ===", max_layer);

                kv_cache.clear();
                let mut forward_pool =
                    ForwardBufferPool::new(device.clone(), reuse_pool.clone());
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                let mut current_input = hidden_buffer.clone();

                for layer_idx in 0..=max_layer {
                    let output = model
                        .forward_layer_gpu_encode(
                            &model.layers[layer_idx],
                            layer_idx,
                            &current_input,
                            kv_cache,
                            gpu_ctx,
                            &mut forward_pool,
                            encoder,
                        )
                        .expect(&format!("Layer {} failed", layer_idx));

                    current_input = output;
                }

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                // Check all layer outputs
                println!("After encoding {} layers:", max_layer + 1);
                for layer_idx in 0..=max_layer {
                    let k_cache = kv_cache.get_k(layer_idx, 0);
                    let k_nan = k_cache.iter().filter(|&&v| v.is_nan()).count();
                    if k_nan > 0 {
                        println!(
                            "  Layer {} K cache: {} NaN ✗",
                            layer_idx, k_nan
                        );
                    } else {
                        println!("  Layer {} K cache: 0 NaN ✓", layer_idx);
                    }
                }

                let mut last_output = vec![0.0f32; hidden_size];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        current_input.contents() as *const f32,
                        last_output.as_mut_ptr(),
                        hidden_size,
                    );
                }
                let nan_count =
                    last_output.iter().filter(|&&v| v.is_nan()).count();
                if nan_count > 0 {
                    println!("  Final output: {} NaN ✗\n", nan_count);
                } else {
                    println!("  Final output: 0 NaN ✓\n");
                }

                // Break if NaN detected
                if nan_count > 0 {
                    println!("*** NaN detected at max_layer={} ***", max_layer);
                    break;
                }
            }
        }
    }
}
