// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compare: GPU execution time vs CPU overhead

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
            let n_layers = model.layers.len();
            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            println!("\n=== GPU Execution vs CPU Overhead ===");
            println!("Layers: {}", n_layers);

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

            // Test 1: Measure GPU time only (remove CPU overhead)
            println!("\n--- Test 1: Single CB, measure GPU execution ---");
            let mut layer_input = hidden_buffer.clone();

            let encode_start = Instant::now();
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
            let encode_time = encode_start.elapsed();

            // Use Metal's GPU timing
            let gpu_start = Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let wall_time = gpu_start.elapsed();

            println!("Encoding (CPU): {:?}", encode_time);
            println!("Wall clock (GPU+wait): {:?}", wall_time);

            // Test 2: Measure each operation with GPU timing
            println!("\n--- Test 2: GPU time for each kernel type ---");

            // Reset
            kv_cache.clear();
            forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            layer_input = hidden_buffer.clone();

            // Track total GPU time
            let mut total_gpu_time = std::time::Duration::ZERO;

            for layer_idx in 0..n_layers {
                let layer = &model.layers[layer_idx];

                // Measure this layer's GPU execution
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();

                let layer_output = model
                    .forward_layer_gpu_encode(
                        layer,
                        layer_idx,
                        &layer_input,
                        kv_cache,
                        gpu_ctx,
                        &mut forward_pool,
                        encoder,
                    )
                    .expect(&format!("Layer {} failed", layer_idx));

                encoder.end_encoding();

                // Time just this layer's execution
                let start = Instant::now();
                cb.commit();
                cb.wait_until_completed();
                let layer_time = start.elapsed();

                total_gpu_time += layer_time;
                layer_input = layer_output;

                if layer_idx < 5 {
                    println!("Layer {}: {:?}", layer_idx, layer_time);
                }
            }

            println!("...");
            println!("Total (28 layers, separate CBs): {:?}", total_gpu_time);
            println!("Average per layer: {:?}", total_gpu_time / 28);

            // Compare
            println!("\n=== Comparison ===");
            println!("Single CB (all 28 layers): {:?}", wall_time);
            println!("28 separate CBs: {:?}", total_gpu_time);
            println!(
                "Ratio: {:.1}x",
                total_gpu_time.as_secs_f64() / wall_time.as_secs_f64()
            );
        }
    }
}
