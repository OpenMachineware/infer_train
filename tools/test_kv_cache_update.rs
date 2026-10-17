// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug KV cache update

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
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
            use metal::MTLResourceOptions;
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            use infer_train::buffer::ForwardBufferPool;
            use std::cell::RefCell;
            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            // Get KV cache for layer 0 BEFORE forward (using proper F16->F32 conversion)
            let k_before = kv_cache.get_k(0, 0);
            println!(
                "K cache BEFORE forward (pos 0): first 5: {:?}",
                &k_before[..5.min(k_before.len())]
            );

            // Forward Layer 0
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

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

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Check Layer 0 output
            let mut layer0_data = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    layer0_out.contents() as *const f32,
                    layer0_data.as_mut_ptr(),
                    hidden_size,
                );
            }
            let num_nan = layer0_data.iter().filter(|&&v| v.is_nan()).count();
            println!(
                "\nLayer 0 output: {} NaN, first 5: {:?}",
                num_nan,
                &layer0_data[..5]
            );

            // Check KV cache AFTER forward (using proper F16->F32 conversion)
            let k_after = kv_cache.get_k(0, 0);
            let k_nan = k_after.iter().filter(|&&v| v.is_nan()).count();
            let k_zero = k_after.iter().filter(|&&v| v == 0.0).count();
            println!(
                "\nK cache AFTER forward (pos 0): {} NaN, {} zero, first 5: {:?}",
                k_nan,
                k_zero,
                &k_after[..5.min(k_after.len())]
            );
        }
    }
}
