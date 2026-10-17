// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug GPU forward pass step by step

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    // Load tokenizer
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

    // Load model
    println!("Loading model: {}", model_path);
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");
    println!("GPU context available: {}", ctx.gpu_ctx.is_some());
    println!("GPU KV cache available: {}", ctx.gpu_kv_cache.is_some());

    // Test single token
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt);
    println!("\nPrompt: \"{}\"", prompt);
    println!("Tokens: {:?}", tokens);

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (ctx.gpu_ctx.as_ref(), ctx.gpu_kv_cache.as_mut())
        {
            let model = &ctx.model;
            let hidden_size = model.config.hidden_size;

            // Test layer 0 only
            println!("\n=== Test Layer 0 Forward ===");

            let layer0 = &model.layers[0];

            // Get embedding
            let hidden =
                model.embedding_lookup_impl(tokens[0]).expect("No embedding");
            println!(
                "Embedding: {} values, first 5: {:?}",
                hidden.len(),
                &hidden[..5.min(hidden.len())]
            );

            // Upload to GPU
            use metal::MTLResourceOptions;
            let device = gpu_ctx.device();
            let hidden_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            // Create forward pool
            use infer_train::buffer::ForwardBufferPool;
            use std::cell::RefCell;
            use std::rc::Rc;
            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            // Create command buffer
            let queue = gpu_ctx.queue();
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Forward layer 0
            let result = model.forward_layer_gpu_encode(
                layer0,
                0,
                &hidden_buffer,
                kv_cache,
                gpu_ctx,
                &mut forward_pool,
                encoder,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            match result {
                Some(output_buffer) => {
                    // Read output
                    let mut output = vec![0.0f32; hidden_size];
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            output_buffer.contents() as *const f32,
                            output.as_mut_ptr(),
                            hidden_size,
                        );
                    }

                    let num_nan =
                        output.iter().filter(|&&v| v.is_nan()).count();
                    let num_zero = output.iter().filter(|&&v| v == 0.0).count();
                    let min =
                        output.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = output
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);

                    println!(
                        "Layer 0 output: {} values, {} NaN, {} zero, range [{:.4}, {:.4}]",
                        output.len(),
                        num_nan,
                        num_zero,
                        min,
                        max
                    );
                    println!("First 5: {:?}", &output[..5.min(output.len())]);
                }
                None => {
                    println!("Layer 0 forward returned None");
                }
            }
        }
    }
}
