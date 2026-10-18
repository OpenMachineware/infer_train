// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark: GPU prefill throughput vs llama.cpp

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

    // Load tokenizer
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

    // Use a longer prompt for proper prefill benchmark
    let prompt = "The quick brown fox jumps over the lazy dog. This is a test of the GPU prefill performance. We want to see how fast we can process multiple tokens through the model.";
    let tokens = tokenizer.encode(prompt);
    println!("Prompt: \"{}\"", prompt);
    println!("Tokens: {} tokens", tokens.len());

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

            println!("\n=== GPU Prefill Benchmark ===");
            println!("Layers: {}", n_layers);
            println!("Hidden size: {}", hidden_size);
            println!("Tokens to process: {}", tokens.len());

            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            // Process all tokens through the model
            let total_start = Instant::now();
            let mut final_layer_input = None;

            for (token_idx, &token) in tokens.iter().enumerate() {
                let mut forward_pool =
                    ForwardBufferPool::new(device.clone(), reuse_pool.clone());

                // Get embedding for this token
                let hidden =
                    model.embedding_lookup_impl(token).expect("No embedding");
                let hidden_buffer = Rc::new(device.new_buffer_with_data(
                    hidden.as_ptr() as *const std::ffi::c_void,
                    (hidden_size * 4) as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                ));

                // Process through all layers
                let mut layer_input = hidden_buffer.clone();

                for layer_idx in 0..n_layers {
                    let command_buffer = queue.new_command_buffer();
                    let encoder = command_buffer.new_compute_command_encoder();

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
                    command_buffer.commit();
                    command_buffer.wait_until_completed();

                    layer_input = layer_output;
                }

                // Save final layer output for this token
                final_layer_input = Some(layer_input);

                // Move to next position in KV cache
                kv_cache.advance();

                if (token_idx + 1) % 5 == 0 {
                    println!(
                        "Processed {}/{} tokens",
                        token_idx + 1,
                        tokens.len()
                    );
                }
            }

            let total_time = total_start.elapsed();
            let tokens_per_sec = tokens.len() as f64 / total_time.as_secs_f64();

            println!("\n=== Results ===");
            println!("Total tokens: {}", tokens.len());
            println!("Total time: {:?}", total_time);
            println!("Throughput: {:.1} t/s", tokens_per_sec);

            // Check for NaN in final output
            if let Some(layer_input) = final_layer_input {
                let contents = layer_input.contents() as *const f32;
                let num_nan = unsafe {
                    (0..hidden_size)
                        .filter(|&i| (*contents.add(i)).is_nan())
                        .count()
                };

                if num_nan > 0 {
                    println!("WARNING: {} NaN in final output", num_nan);
                }
            }
        }
    }
}
