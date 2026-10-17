// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug full GPU forward pass - check final hidden state

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

    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

    println!("Loading model: {}", model_path);
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");
    println!("Layers: {}", ctx.model.layers.len());
    println!("GPU context available: {}", ctx.gpu_ctx.is_some());

    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt);
    println!("\nPrompt: \"{}\", tokens: {:?}", prompt, tokens);

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let (Some(ref gpu_ctx), Some(ref mut kv_cache)) =
            (ctx.gpu_ctx.as_ref(), ctx.gpu_kv_cache.as_mut())
        {
            let model = &ctx.model;
            let hidden_size = model.config.hidden_size;
            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            // Get embedding
            let hidden =
                model.embedding_lookup_impl(tokens[0]).expect("No embedding");

            use metal::MTLResourceOptions;
            let hidden_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            // Forward all layers
            use infer_train::buffer::ForwardBufferPool;
            use std::cell::RefCell;
            use std::rc::Rc;
            let reuse_pool = Rc::new(RefCell::new(Vec::new()));

            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());
            let mut current_buffer = hidden_buffer.clone();

            for (layer_idx, layer) in model.layers.iter().enumerate() {
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

            // Final norm
            let final_norm = model.norm.as_ref().expect("No final norm");
            let final_norm_buffer = gpu_ctx.get_or_create_buffer(final_norm);
            let final_hidden = forward_pool.get(hidden_size * 4);

            gpu_ctx.rms_norm_encode(
                encoder,
                &current_buffer,
                &final_norm_buffer,
                &final_hidden,
                1,
                hidden_size,
                1e-5,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Check final hidden
            let mut final_hidden_data = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    final_hidden.contents() as *const f32,
                    final_hidden_data.as_mut_ptr(),
                    hidden_size,
                );
            }

            let num_nan =
                final_hidden_data.iter().filter(|&&v| v.is_nan()).count();
            let num_zero =
                final_hidden_data.iter().filter(|&&v| v == 0.0).count();
            let min =
                final_hidden_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = final_hidden_data
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            println!(
                "\nFinal hidden (after norm): {} NaN, {} zero, range [{:.4}, {:.4}]",
                num_nan, num_zero, min, max
            );
            println!("First 5: {:?}", &final_hidden_data[..5]);

            // Check output tensor
            if let Some(output) = &model.output {
                println!(
                    "\nOutput tensor: shape={:?}, dtype={:?}",
                    output.shape, output.dtype
                );

                // Try CPU GEMV
                println!("\n=== CPU GEMV on final hidden ===");
                let vocab_size = if output.shape.len() == 2
                    && output.shape[1] == hidden_size
                {
                    output.shape[0] // transposed
                } else {
                    output.shape[0]
                };
                println!("Vocab size: {}", vocab_size);

                let logits = model.gemv(&final_hidden_data, output, vocab_size);

                match logits {
                    Some(l) => {
                        let num_zero = l.iter().filter(|&&v| v == 0.0).count();
                        let max =
                            l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        println!(
                            "Logits: {} values, {} zero, max={:.4}",
                            l.len(),
                            num_zero,
                            max
                        );
                    }
                    None => println!("CPU GEMV failed"),
                }
            }
        }
    }
}
