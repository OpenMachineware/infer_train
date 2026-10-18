// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test: fewer command buffers (llama.cpp uses 1-2, not 28)

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

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
    let tokens = tokenizer.encode("Hello");
    println!("Tokens: {:?}", tokens);

    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (ctx.gpu_ctx.as_ref(), ctx.gpu_kv_cache.as_mut())
        {
            let model = &ctx.model;
            let hidden_size = model.config.hidden_size;
            let n_layers = model.layers.len();
            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            println!("\n=== Test: Fewer command buffers ===");
            println!("Layers: {}", n_layers);

            // Get embedding for first token
            let hidden =
                model.embedding_lookup_impl(tokens[0]).expect("No embedding");
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            // Test 1: Single command buffer for all layers
            println!("\n--- Test 1: Single command buffer for all layers ---");
            {
                let mut forward_pool =
                    ForwardBufferPool::new(device.clone(), reuse_pool.clone());
                let mut layer_input = hidden_buffer.clone();

                let start = Instant::now();
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                for layer_idx in 0..n_layers {
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

                    layer_input = layer_output;
                }

                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();

                let elapsed = start.elapsed();
                println!("Time: {:?}", elapsed);

                // Check for NaN
                let contents = layer_input.contents() as *const f32;
                let num_nan = unsafe {
                    (0..hidden_size)
                        .filter(|&i| (*contents.add(i)).is_nan())
                        .count()
                };
                println!("NaN: {}", num_nan);

                if num_nan == 0 {
                    let tps = 1.0 / elapsed.as_secs_f64();
                    println!("Throughput: {:.1} t/s", tps);
                }
            }

            // Reset KV cache
            kv_cache.clear();

            // Test 2: Two command buffers (half layers each)
            println!("\n--- Test 2: Two command buffers (14 layers each) ---");
            {
                let mut forward_pool =
                    ForwardBufferPool::new(device.clone(), reuse_pool.clone());
                let mut layer_input = hidden_buffer.clone();

                let start = Instant::now();

                // First half
                let cb1 = queue.new_command_buffer();
                let encoder1 = cb1.new_compute_command_encoder();
                for layer_idx in 0..n_layers / 2 {
                    let layer_output = model
                        .forward_layer_gpu_encode(
                            &model.layers[layer_idx],
                            layer_idx,
                            &layer_input,
                            kv_cache,
                            gpu_ctx,
                            &mut forward_pool,
                            encoder1,
                        )
                        .expect(&format!("Layer {} failed", layer_idx));
                    layer_input = layer_output;
                }
                encoder1.end_encoding();
                cb1.commit();

                // Second half
                let cb2 = queue.new_command_buffer();
                let encoder2 = cb2.new_compute_command_encoder();
                for layer_idx in n_layers / 2..n_layers {
                    let layer_output = model
                        .forward_layer_gpu_encode(
                            &model.layers[layer_idx],
                            layer_idx,
                            &layer_input,
                            kv_cache,
                            gpu_ctx,
                            &mut forward_pool,
                            encoder2,
                        )
                        .expect(&format!("Layer {} failed", layer_idx));
                    layer_input = layer_output;
                }
                encoder2.end_encoding();
                cb2.commit();

                // Wait for both
                cb1.wait_until_completed();
                cb2.wait_until_completed();

                let elapsed = start.elapsed();
                println!("Time: {:?}", elapsed);

                // Check for NaN
                let contents = layer_input.contents() as *const f32;
                let num_nan = unsafe {
                    (0..hidden_size)
                        .filter(|&i| (*contents.add(i)).is_nan())
                        .count()
                };
                println!("NaN: {}", num_nan);

                if num_nan == 0 {
                    let tps = 1.0 / elapsed.as_secs_f64();
                    println!("Throughput: {:.1} t/s", tps);
                }
            }
        }
    }
}
