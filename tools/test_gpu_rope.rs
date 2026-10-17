// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test RoPE kernel

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
    let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt);
    let token = tokens[0];
    println!("\nToken: {}", token);

    let hidden =
        ctx.model.embedding_lookup_impl(token).expect("Embedding failed");
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

    let head_dim = ctx.model.config.head_dim;
    let n_heads = cpu_q.len() / head_dim;

    // CPU RoPE
    let cpu_q_rope = ctx
        .model
        .apply_rope(&cpu_q, 0, n_heads, head_dim)
        .expect("RoPE failed");
    println!(
        "CPU Q RoPE: len={}, min={:.4}, max={:.4}, nan={}",
        cpu_q_rope.len(),
        cpu_q_rope.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_q_rope.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        cpu_q_rope.iter().filter(|&&v| v.is_nan()).count()
    );

    // GPU RoPE
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let Some(ref gpu_ctx) = ctx.gpu_ctx {
            use metal::*;

            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            let q_buffer_rc = std::rc::Rc::new(device.new_buffer_with_data(
                cpu_q.as_ptr() as *const std::ffi::c_void,
                (cpu_q.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            let q_rope_buffer_rc = std::rc::Rc::new(device.new_buffer(
                (cpu_q_rope.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            gpu_ctx.rope_neox_encode(
                encoder,
                &q_buffer_rc,
                &[0],
                1,
                n_heads,
                head_dim,
                10000.0,
                1.0,
                &q_rope_buffer_rc,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let gpu_q_rope: Vec<f32> = unsafe {
                std::slice::from_raw_parts(
                    q_rope_buffer_rc.contents() as *const f32,
                    cpu_q_rope.len(),
                )
                .to_vec()
            };

            println!(
                "GPU Q RoPE: len={}, min={:.4}, max={:.4}, nan={}",
                gpu_q_rope.len(),
                gpu_q_rope.iter().cloned().fold(f32::INFINITY, f32::min),
                gpu_q_rope.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                gpu_q_rope.iter().filter(|&&v| v.is_nan()).count()
            );

            let max_diff = cpu_q_rope
                .iter()
                .zip(gpu_q_rope.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("Max diff: {:.6}", max_diff);

            // Check first 10 values
            println!("\nFirst 10 values comparison:");
            for i in 0..10 {
                println!(
                    "  [{}]: CPU={:.4}, GPU={:.4}, diff={:.6}",
                    i,
                    cpu_q_rope[i],
                    gpu_q_rope[i],
                    (cpu_q_rope[i] - gpu_q_rope[i]).abs()
                );
            }
        }
    }
}
