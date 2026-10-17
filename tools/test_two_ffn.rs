// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test if FFN causes issues when run twice

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
            let w1 = layer0.w1.as_ref().unwrap();
            let w2 = layer0.w2.as_ref().unwrap();
            let w3 = layer0.w3.as_ref().unwrap();
            let intermediate_size = w1.shape[1];

            println!("=== Test: FFN twice in one command buffer ===\n");

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let input = forward_pool.register(hidden_buffer);

            // FFN 1
            let gate1 = forward_pool.get_zeroed(intermediate_size * 4);
            let up1 = forward_pool.get_zeroed(intermediate_size * 4);
            let silu1 = forward_pool.get_zeroed(intermediate_size * 4);
            let hidden_ffn1 = forward_pool.get_zeroed(intermediate_size * 4);
            let ffn_out1 = forward_pool.get_zeroed(hidden_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &input,
                    w1,
                    intermediate_size,
                    gpu_ctx,
                    &gate1,
                )
                .unwrap();
            model
                .gemv_encode(
                    encoder,
                    &input,
                    w3,
                    intermediate_size,
                    gpu_ctx,
                    &up1,
                )
                .unwrap();

            gpu_ctx.silu_encode(encoder, &gate1, &silu1, intermediate_size);
            gpu_ctx.mul_encode(
                encoder,
                &silu1,
                &up1,
                &hidden_ffn1,
                intermediate_size,
            );
            model
                .gemv_encode(
                    encoder,
                    &hidden_ffn1,
                    w2,
                    hidden_size,
                    gpu_ctx,
                    &ffn_out1,
                )
                .unwrap();

            println!("FFN 1 encoded");

            // FFN 2 (using ffn_out1 as input)
            let input2 = forward_pool.register(ffn_out1.clone());

            let gate2 = forward_pool.get_zeroed(intermediate_size * 4);
            let up2 = forward_pool.get_zeroed(intermediate_size * 4);
            let silu2 = forward_pool.get_zeroed(intermediate_size * 4);
            let hidden_ffn2 = forward_pool.get_zeroed(intermediate_size * 4);
            let ffn_out2 = forward_pool.get_zeroed(hidden_size * 4);

            model
                .gemv_encode(
                    encoder,
                    &input2,
                    w1,
                    intermediate_size,
                    gpu_ctx,
                    &gate2,
                )
                .unwrap();
            model
                .gemv_encode(
                    encoder,
                    &input2,
                    w3,
                    intermediate_size,
                    gpu_ctx,
                    &up2,
                )
                .unwrap();

            gpu_ctx.silu_encode(encoder, &gate2, &silu2, intermediate_size);
            gpu_ctx.mul_encode(
                encoder,
                &silu2,
                &up2,
                &hidden_ffn2,
                intermediate_size,
            );
            model
                .gemv_encode(
                    encoder,
                    &hidden_ffn2,
                    w2,
                    hidden_size,
                    gpu_ctx,
                    &ffn_out2,
                )
                .unwrap();

            println!("FFN 2 encoded");

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            println!("\n=== Results ===\n");
            check_buffer("ffn_out1", &ffn_out1);
            check_buffer("ffn_out2", &ffn_out2);
            check_buffer("gate1", &gate1);
            check_buffer("gate2", &gate2);

            let contents1 = ffn_out1.contents() as *const f32;
            let len1 = ffn_out1.length() as usize / 4;
            let nan1 = unsafe {
                (0..len1).filter(|&i| (*contents1.add(i)).is_nan()).count()
            };

            if nan1 > 0 {
                println!("\n*** FFN has issues! ***");
            } else {
                println!("\n*** FFN works fine! ***");
            }
        }
    }
}
