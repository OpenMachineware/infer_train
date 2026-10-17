// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Minimal test: only layer 0 attention norm

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;
use infer_train::tokenize::BpeTokenizer;
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

    // Load model
    println!("Loading model: {}", model_path);
    let start = Instant::now();
    let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");
    println!("Model loaded in {:?}", start.elapsed());

    // Test one token
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt);
    let token = tokens[0];
    println!("\nToken: {}", token);

    // Test embedding lookup
    let hidden =
        ctx.model.embedding_lookup_impl(token).expect("Embedding failed");
    println!(
        "Embedding: len={}, min={:.4}, max={:.4}",
        hidden.len(),
        hidden.iter().cloned().fold(f32::INFINITY, f32::min),
        hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Test layer 0 attention norm
    let layer0 = &ctx.model.layers[0];
    let attn_norm = layer0.attention_norm.as_ref().expect("No attention norm");

    // CPU version
    let cpu_normed = ctx
        .model
        .apply_rms_norm(&hidden, attn_norm, hidden.len())
        .expect("RMS norm failed");
    println!(
        "CPU normed: len={}, min={:.4}, max={:.4}, nan={}",
        cpu_normed.len(),
        cpu_normed.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        cpu_normed.iter().filter(|&&v| v.is_nan()).count()
    );

    // GPU version
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let Some(ref gpu_ctx) = ctx.gpu_ctx {
            use metal::*;

            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            // Upload embedding
            let input_buffer = device.new_buffer_with_data(
                hidden.as_ptr() as *const std::ffi::c_void,
                (hidden.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Create output buffer
            let output_buffer = device.new_buffer(
                (hidden.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Create command buffer
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Get norm weight buffer
            let norm_weight_buffer = gpu_ctx.get_or_create_buffer(attn_norm);

            // Encode RMS norm
            gpu_ctx.rms_norm_encode(
                encoder,
                &input_buffer,
                &norm_weight_buffer,
                &output_buffer,
                1,
                hidden.len(),
                1e-5,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read result
            let gpu_normed: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    output_buffer.contents() as *const f32,
                    hidden.len(),
                )
                .to_vec()
            };

            println!(
                "GPU normed: len={}, min={:.4}, max={:.4}, nan={}",
                gpu_normed.len(),
                gpu_normed.iter().cloned().fold(f32::INFINITY, f32::min),
                gpu_normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                gpu_normed.iter().filter(|&&v| v.is_nan()).count()
            );

            // Compare
            let max_diff = cpu_normed
                .iter()
                .zip(gpu_normed.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("Max diff: {:.6}", max_diff);
        }
    }
}
