// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test: each layer uses independent command buffer (llama.cpp approach)

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

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

    // Load tokenizer like test_new_gpu_path
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");
    let tokens = tokenizer.encode("Hello");
    println!("Tokens: {:?}", tokens);

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    println!("After model loading - checking cfg condition");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("Model loaded");
        println!("GPU context available: {}", ctx.gpu_ctx.is_some());
        println!("GPU KV cache available: {}", ctx.gpu_kv_cache.is_some());
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (ctx.gpu_ctx.as_ref(), ctx.gpu_kv_cache.as_mut())
        {
            let model = &ctx.model;
            let hidden_size = model.config.hidden_size;
            let n_layers = model.layers.len();
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

            println!(
                "=== Test: Each layer uses independent command buffer ===\n"
            );
            println!(
                "Approach: llama.cpp splits graph into chunks per command buffer\n"
            );
            println!("Model has {} layers", n_layers);
            println!("Hidden size: {}", hidden_size);
            eprintln!("Starting forward pass...");

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            // Collect command buffers for later synchronization
            // Process layers one by one, each with its own command buffer
            let mut layer_input = hidden_buffer.clone();

            let forward_start = Instant::now();
            for layer_idx in 0..n_layers {
                // Create new command buffer for this layer
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                // Encode layer
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
                    .expect(&format!("Layer {} failed", layer_idx));

                encoder.end_encoding();

                // Commit immediately (llama.cpp commits each buffer right after encoding)
                command_buffer.commit();
                // Wait immediately instead of collecting all command buffers
                command_buffer.wait_until_completed();
                // command_buffers.push(command_buffer); // Not needed if we wait
                // print!("."); // Progress indicator

                // println!("Layer {} encoded and committed", layer_idx);

                // Use this layer's output as next layer's input
                layer_input = layer_output;
            }

            let forward_time = forward_start.elapsed();
            println!("\nForward pass time: {:?}", forward_time);

            // Check results
            println!("\n=== Results ===\n");

            for layer_idx in 0..n_layers {
                let k_cache = kv_cache.get_k(layer_idx, 0);
                let k_nan = k_cache.iter().filter(|&&v| v.is_nan()).count();
                println!("Layer {} K cache: {} NaN", layer_idx, k_nan);
            }

            // Check final layer output
            let contents = layer_input.contents() as *const f32;
            let num_nan = unsafe {
                (0..hidden_size)
                    .filter(|&i| (*contents.add(i)).is_nan())
                    .count()
            };
            println!("\nFinal output: {} NaN", num_nan);

            if num_nan == 0 {
                println!(
                    "\n*** SUCCESS: Each layer in separate command buffer works! ***"
                );
            } else {
                println!("\n*** Still has NaN issue ***");
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!(
            "GPU support not available (not aarch64 or metal feature disabled)"
        );
    }
}
