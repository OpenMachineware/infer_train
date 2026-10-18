// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test single command buffer with layer-by-layer output verification

use infer_train::inference::{Device, InferenceContext};
use std::rc::Rc;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    // Test 1: Single command buffer approach
    println!("=== Test 1: Single Command Buffer ===\n");
    {
        let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load model");

        let model = &ctx.model;
        let gpu_ctx = ctx.gpu_ctx.as_ref().unwrap();
        let kv_cache = ctx.gpu_kv_cache.as_mut().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();
        let hidden_size = model.config.hidden_size;

        // Get token embedding
        let hidden = model.embedding_lookup_impl(1).unwrap();
        let hidden_buffer = Rc::new(device.new_buffer_with_data(
            hidden.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        println!("Initial hidden: {:?}", &hidden[..5]);

        let reuse_pool = Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut forward_pool = infer_train::buffer::ForwardBufferPool::new(
            device.clone(),
            reuse_pool.clone(),
        );

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let mut layer_input = hidden_buffer.clone();
        let mut layer_outputs: Vec<Rc<metal::Buffer>> = Vec::new();

        for layer_idx in 0..3.min(model.layers.len()) {
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
                .unwrap();

            layer_outputs.push(layer_output.clone());

            layer_input = layer_output;
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Check each layer output
        for (layer_idx, output_buf) in layer_outputs.iter().enumerate() {
            let mut output = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    output_buf.contents() as *const f32,
                    output.as_mut_ptr(),
                    hidden_size,
                );
            }

            let nan_count = output.iter().filter(|&&v| v.is_nan()).count();
            let max_val = output
                .iter()
                .filter(|&&v| !v.is_nan())
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let min_val = output
                .iter()
                .filter(|&&v| !v.is_nan())
                .cloned()
                .fold(f32::INFINITY, f32::min);

            println!("\nLayer {} output stats:", layer_idx);
            println!("  NaN count: {}", nan_count);
            println!("  Max: {}", max_val);
            println!("  Min: {}", min_val);
            println!("  First 5: {:?}", &output[..5]);
        }
    }

    // Test 2: Per-layer command buffers (with FRESH context)
    println!("\n\n=== Test 2: Per-Layer Command Buffers ===\n");
    {
        let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load model");

        let model = &ctx.model;
        let gpu_ctx = ctx.gpu_ctx.as_ref().unwrap();
        let kv_cache = ctx.gpu_kv_cache.as_mut().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();
        let hidden_size = model.config.hidden_size;

        // Get token embedding (same token)
        let hidden = model.embedding_lookup_impl(1).unwrap();
        let hidden_buffer = Rc::new(device.new_buffer_with_data(
            hidden.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let reuse_pool = Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut forward_pool = infer_train::buffer::ForwardBufferPool::new(
            device.clone(),
            reuse_pool.clone(),
        );

        let mut layer_input = hidden_buffer.clone();

        for layer_idx in 0..3.min(model.layers.len()) {
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();

            let layer_output = model
                .forward_layer_gpu_encode(
                    &model.layers[layer_idx],
                    layer_idx,
                    &layer_input,
                    kv_cache,
                    gpu_ctx,
                    &mut forward_pool,
                    enc,
                )
                .unwrap();

            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            // Check this layer's output
            let mut output = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    layer_output.contents() as *const f32,
                    output.as_mut_ptr(),
                    hidden_size,
                );
            }

            let nan_count = output.iter().filter(|&&v| v.is_nan()).count();
            println!("Layer {} NaN count: {}", layer_idx, nan_count);

            layer_input = layer_output;
        }

        // Final output stats
        let mut output = vec![0.0f32; hidden_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                layer_input.contents() as *const f32,
                output.as_mut_ptr(),
                hidden_size,
            );
        }

        println!("\nFinal output stats:");
        println!(
            "  NaN count: {}",
            output.iter().filter(|&&v| v.is_nan()).count()
        );
        println!("  First 5: {:?}", &output[..5]);
    }
}
