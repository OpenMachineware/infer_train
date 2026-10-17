// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug Layer 1 forward with detailed check

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
use std::rc::Rc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");
    let tokens = tokenizer.encode("Hello");

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
                model.embedding_lookup_impl(tokens[0]).expect("No embedding");
            println!("Embedding first 5: {:?}", &hidden[..5]);

            use metal::MTLResourceOptions;
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            use infer_train::buffer::ForwardBufferPool;
            use std::cell::RefCell;
            let reuse_pool = Rc::new(RefCell::new(Vec::new()));

            // Layer 0
            println!("\n=== Layer 0 ===");
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
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

            let mut layer0_data = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    layer0_out.contents() as *const f32,
                    layer0_data.as_mut_ptr(),
                    hidden_size,
                );
            }
            println!(
                "Layer 0: {} NaN, {} zero, first 5: {:?}",
                layer0_data.iter().filter(|&&v| v.is_nan()).count(),
                layer0_data.iter().filter(|&&v| v == 0.0).count(),
                &layer0_data[..5]
            );

            kv_cache.advance();

            // Layer 1
            println!("\n=== Layer 1 ===");
            let mut forward_pool2 =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let command_buffer2 = queue.new_command_buffer();
            let encoder2 = command_buffer2.new_compute_command_encoder();

            let layer1_out = model
                .forward_layer_gpu_encode(
                    &model.layers[1],
                    1,
                    &layer0_out,
                    kv_cache,
                    gpu_ctx,
                    &mut forward_pool2,
                    encoder2,
                )
                .expect("Layer 1 failed");

            encoder2.end_encoding();
            command_buffer2.commit();
            command_buffer2.wait_until_completed();

            let mut layer1_data = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    layer1_out.contents() as *const f32,
                    layer1_data.as_mut_ptr(),
                    hidden_size,
                );
            }
            println!(
                "Layer 1: {} NaN, {} zero, first 5: {:?}",
                layer1_data.iter().filter(|&&v| v.is_nan()).count(),
                layer1_data.iter().filter(|&&v| v == 0.0).count(),
                &layer1_data[..5]
            );
        }
    }
}
