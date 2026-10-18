// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test: Why does Layer 3+ take 500ms?

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

            println!("\n=== Test First 3 Layers ===");

            let hidden =
                model.embedding_lookup_impl(tokens[0]).expect("No embedding");
            let hidden_buffer = Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            let reuse_pool = Rc::new(RefCell::new(Vec::new()));
            use infer_train::buffer::ForwardBufferPool;

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            let mut layer_input = hidden_buffer.clone();

            // Test first 3 layers
            for layer_idx in 0..3 {
                let start = Instant::now();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();

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
                cb.commit();
                cb.wait_until_completed();
                let time = start.elapsed();

                println!("Layer {}: {:?}", layer_idx, time);
                layer_input = layer_output;
                kv_cache.advance(); // Move to next position
            }

            println!("\n=== Now test Layer 3 alone ===");
            let start = Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();

            let layer_output = model
                .forward_layer_gpu_encode(
                    &model.layers[3],
                    3,
                    &layer_input,
                    kv_cache,
                    gpu_ctx,
                    &mut forward_pool,
                    encoder,
                )
                .expect("Layer 3 failed");

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let time = start.elapsed();

            println!("Layer 3: {:?}", time);

            // Check KV cache state
            println!("\n=== KV Cache State ===");
            println!("Max seq: {}", kv_cache.max_seq());
            println!("Current seq: {}", kv_cache.seq_len());

            // Check GPU memory
            println!("\n=== GPU Memory ===");
            println!(
                "Current allocated: {} MB",
                device.current_allocated_size() / 1_000_000
            );
        }
    }
}
