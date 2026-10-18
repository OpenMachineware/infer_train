// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile: Time breakdown of GPU forward pass

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

            println!("\n=== GPU Forward Pass Profile ===");
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

            // Profile encoding vs execution
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

            let exec_start = Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let exec_time = exec_start.elapsed();

            println!("\n=== Time Breakdown ===");
            println!("Encoding time: {:?}", encode_time);
            println!("Execution time: {:?}", exec_time);
            println!("Total time: {:?}", encode_time + exec_time);

            // Check for NaN
            let contents = layer_input.contents() as *const f32;
            let num_nan = unsafe {
                (0..hidden_size)
                    .filter(|&i| (*contents.add(i)).is_nan())
                    .count()
            };
            println!("\nNaN: {}", num_nan);

            // Estimate kernel time
            // 7 GEMVs per layer × 28 layers = 196 GEMVs
            // Each GEMV: 1024×4096 or 4096×1024
            // From benchmark: ~400μs per GEMV
            let estimated_gemv = 196.0 * 400.0 / 1000.0; // ms
            println!("\n=== Estimated GEMV time: {:.1}ms ===", estimated_gemv);
        }
    }
}
