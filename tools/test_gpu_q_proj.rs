// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test layer 0 Q projection

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

    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let tokenizer =
        BpeTokenizer::from_gguf(&gguf).expect("Failed to load tokenizer");

    println!("Loading model: {}", model_path);
    let start = Instant::now();
    let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");
    println!("Model loaded in {:?}", start.elapsed());

    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt);
    let token = tokens[0];
    println!("\nToken: {}", token);

    let hidden =
        ctx.model.embedding_lookup_impl(token).expect("Embedding failed");
    println!("Embedding: len={}", hidden.len());

    let layer0 = &ctx.model.layers[0];
    let attn_norm = layer0.attention_norm.as_ref().expect("No attention norm");
    let wq = layer0.wq.as_ref().expect("No wq");

    // CPU version
    let cpu_normed = ctx
        .model
        .apply_rms_norm(&hidden, attn_norm, hidden.len())
        .expect("RMS norm failed");
    let cpu_q =
        ctx.model.gemv(&cpu_normed, wq, wq.shape[1]).expect("GEMV failed");
    println!(
        "CPU Q: len={}, min={:.4}, max={:.4}, nan={}",
        cpu_q.len(),
        cpu_q.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_q.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        cpu_q.iter().filter(|&&v| v.is_nan()).count()
    );

    // GPU version
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let Some(ref gpu_ctx) = ctx.gpu_ctx {
            use metal::*;

            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            let normed_buffer = device.new_buffer_with_data(
                cpu_normed.as_ptr() as *const std::ffi::c_void,
                (cpu_normed.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let q_buffer = device.new_buffer(
                (cpu_q.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            let normed_buffer_rc = std::rc::Rc::new(normed_buffer);
            let q_buffer_rc = std::rc::Rc::new(q_buffer);

            let success = ctx.model.gemv_encode(
                encoder,
                &normed_buffer_rc,
                wq,
                cpu_q.len(),
                gpu_ctx,
                &q_buffer_rc,
            );

            if success.is_none() {
                println!("GPU Q encode failed!");
                return;
            }

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let gpu_q: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    q_buffer_rc.contents() as *const f32,
                    cpu_q.len(),
                )
                .to_vec()
            };

            println!(
                "GPU Q: len={}, min={:.4}, max={:.4}, nan={}",
                gpu_q.len(),
                gpu_q.iter().cloned().fold(f32::INFINITY, f32::min),
                gpu_q.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                gpu_q.iter().filter(|&&v| v.is_nan()).count()
            );

            let max_diff = cpu_q
                .iter()
                .zip(gpu_q.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("Max diff: {:.6}", max_diff);
        }
    }
}
