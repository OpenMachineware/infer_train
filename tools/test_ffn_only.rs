// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test FFN only to isolate NaN issue

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

            // Get embedding
            let hidden =
                model.embedding_lookup_impl(9707).expect("No embedding");
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            // Run attention and residual add to get the input to FFN
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let layer0 = &model.layers[0];

            // Just run RMS norm to get a reasonable input for FFN
            let norm_weight_buffer =
                gpu_ctx.get_or_create_buffer(layer0.ffn_norm.as_ref().unwrap());
            let ffn_input = forward_pool.get_zeroed(hidden_size * 4);
            gpu_ctx.rms_norm_encode(
                encoder,
                &hidden_buffer,
                &norm_weight_buffer,
                &ffn_input,
                1,
                hidden_size,
                1e-5,
            );

            // Now test FFN only
            let w1 = layer0.w1.as_ref().expect("No w1");
            let w2 = layer0.w2.as_ref().expect("No w2");
            let w3 = layer0.w3.as_ref().expect("No w3");
            let intermediate_size = w1.shape[1];

            let gate_buffer = forward_pool.get_zeroed(intermediate_size * 4);
            let up_buffer = forward_pool.get_zeroed(intermediate_size * 4);
            let silu_buffer = forward_pool.get_zeroed(intermediate_size * 4);
            let hidden_ffn_buffer =
                forward_pool.get_zeroed(intermediate_size * 4);
            let ffn_out_buffer = forward_pool.get_zeroed(hidden_size * 4);

            // W1 (gate)
            model
                .gemv_encode(
                    encoder,
                    &ffn_input,
                    w1,
                    intermediate_size,
                    gpu_ctx,
                    &gate_buffer,
                )
                .expect("W1 failed");

            // W3 (up)
            model
                .gemv_encode(
                    encoder,
                    &ffn_input,
                    w3,
                    intermediate_size,
                    gpu_ctx,
                    &up_buffer,
                )
                .expect("W3 failed");

            // SiLU(gate)
            gpu_ctx.silu_encode(
                encoder,
                &gate_buffer,
                &silu_buffer,
                intermediate_size,
            );

            // silu * up
            gpu_ctx.mul_encode(
                encoder,
                &silu_buffer,
                &up_buffer,
                &hidden_ffn_buffer,
                intermediate_size,
            );

            // W2 (down projection)
            model
                .gemv_encode(
                    encoder,
                    &hidden_ffn_buffer,
                    w2,
                    hidden_size,
                    gpu_ctx,
                    &ffn_out_buffer,
                )
                .expect("W2 failed");

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Check results
            let mut gate_data = vec![0.0f32; intermediate_size];
            let mut up_data = vec![0.0f32; intermediate_size];
            let mut silu_data = vec![0.0f32; intermediate_size];
            let mut hidden_ffn_data = vec![0.0f32; intermediate_size];
            let mut ffn_out_data = vec![0.0f32; hidden_size];

            unsafe {
                std::ptr::copy_nonoverlapping(
                    gate_buffer.contents() as *const f32,
                    gate_data.as_mut_ptr(),
                    intermediate_size,
                );
                std::ptr::copy_nonoverlapping(
                    up_buffer.contents() as *const f32,
                    up_data.as_mut_ptr(),
                    intermediate_size,
                );
                std::ptr::copy_nonoverlapping(
                    silu_buffer.contents() as *const f32,
                    silu_data.as_mut_ptr(),
                    intermediate_size,
                );
                std::ptr::copy_nonoverlapping(
                    hidden_ffn_buffer.contents() as *const f32,
                    hidden_ffn_data.as_mut_ptr(),
                    intermediate_size,
                );
                std::ptr::copy_nonoverlapping(
                    ffn_out_buffer.contents() as *const f32,
                    ffn_out_data.as_mut_ptr(),
                    hidden_size,
                );
            }

            println!(
                "Gate: {} NaN, first 5: {:?}",
                gate_data.iter().filter(|&&v| v.is_nan()).count(),
                &gate_data[..5.min(gate_data.len())]
            );
            println!(
                "Up: {} NaN, first 5: {:?}",
                up_data.iter().filter(|&&v| v.is_nan()).count(),
                &up_data[..5.min(up_data.len())]
            );
            println!(
                "SiLU: {} NaN, first 5: {:?}",
                silu_data.iter().filter(|&&v| v.is_nan()).count(),
                &silu_data[..5.min(silu_data.len())]
            );
            println!(
                "Hidden FFN: {} NaN, first 5: {:?}",
                hidden_ffn_data.iter().filter(|&&v| v.is_nan()).count(),
                &hidden_ffn_data[..5.min(hidden_ffn_data.len())]
            );
            println!(
                "FFN out: {} NaN, first 5: {:?}",
                ffn_out_data.iter().filter(|&&v| v.is_nan()).count(),
                &ffn_out_data[..5.min(ffn_out_data.len())]
            );

            // Check NaN positions in FFN out
            let nan_positions: Vec<usize> = ffn_out_data
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_nan())
                .map(|(i, _)| i)
                .collect();
            println!(
                "FFN out NaN positions: {:?}",
                &nan_positions[..10.min(nan_positions.len())]
            );
        }
    }
}
