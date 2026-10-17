// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test exact pattern of forward_layer_gpu_encode

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::LlamaModel;
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

            let hidden =
                model.embedding_lookup_impl(9707).expect("No embedding");
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            // Use forward_layer_gpu_encode EXACTLY as in test-chain-forward
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Layer 0
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
            println!("Layer 0 encoded via forward_layer_gpu_encode");
            println!(
                "Pool has {} active buffers after Layer 0",
                forward_pool.stats().active
            );

            // Layer 1 - THIS IS THE EXACT PATTERN THAT FAILS
            let layer1_out = model
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
            println!("Layer 1 encoded via forward_layer_gpu_encode");
            println!(
                "Pool has {} active buffers after Layer 1",
                forward_pool.stats().active
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Check results
            let mut layer0_data = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    layer0_out.contents() as *const f32,
                    layer0_data.as_mut_ptr(),
                    hidden_size,
                );
            }
            let layer0_nan =
                layer0_data.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "\nLayer 0 output: {} NaN, first 5: {:?}",
                layer0_nan,
                &layer0_data[..5.min(layer0_data.len())]
            );

            // Print NaN positions
            let nan_positions: Vec<usize> = layer0_data
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_nan())
                .map(|(i, _)| i)
                .collect();
            println!(
                "Layer 0 NaN positions: {:?}",
                &nan_positions[..10.min(nan_positions.len())]
            );

            let mut layer1_data = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    layer1_out.contents() as *const f32,
                    layer1_data.as_mut_ptr(),
                    hidden_size,
                );
            }
            let layer1_nan =
                layer1_data.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "Layer 1 output: {} NaN, first 5: {:?}",
                layer1_nan,
                &layer1_data[..5.min(layer1_data.len())]
            );

            let k_cache0 = kv_cache.get_k(0, 0);
            let k_cache1 = kv_cache.get_k(1, 0);
            println!(
                "\nK cache 0: first 5: {:?}",
                &k_cache0[..5.min(k_cache0.len())]
            );
            println!(
                "K cache 1: first 5: {:?}",
                &k_cache1[..5.min(k_cache1.len())]
            );
        }
    }
}
