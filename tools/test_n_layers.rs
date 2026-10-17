// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test first N layers to isolate the issue

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
use std::rc::Rc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <num_layers>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let num_layers: usize = args[2].parse().expect("Invalid num_layers");

    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

    println!("Loading model: {}", model_path);
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");
    println!("Total layers: {}", ctx.model.layers.len());
    println!("Testing {} layers", num_layers);

    let tokens = tokenizer.encode("Hello");
    println!("Tokens: {:?}", tokens);

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
            println!("Embedding: first 5: {:?}", &hidden[..5]);

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

            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let mut current_buffer = hidden_buffer.clone();
            let layers_to_test = num_layers.min(model.layers.len());

            for layer_idx in 0..layers_to_test {
                let layer = &model.layers[layer_idx];
                current_buffer = model
                    .forward_layer_gpu_encode(
                        layer,
                        layer_idx,
                        &current_buffer,
                        kv_cache,
                        gpu_ctx,
                        &mut forward_pool,
                        encoder,
                    )
                    .expect("Layer forward failed");
            }

            // Commit and read directly from current_buffer
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let mut output = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    current_buffer.contents() as *const f32,
                    output.as_mut_ptr(),
                    hidden_size,
                );
            }

            let num_zero = output.iter().filter(|&&v| v == 0.0).count();
            let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            println!(
                "Output after {} layers: {} NaN, {} zero, range [{:.4}, {:.4}]",
                layers_to_test,
                output.iter().filter(|&&v| v.is_nan()).count(),
                num_zero,
                min,
                max
            );
            println!("First 5: {:?}", &output[..5]);
        }
    }
}
