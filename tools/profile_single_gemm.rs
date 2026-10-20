// Profile single GEMM operation
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::inference::InferenceContext;
use std::rc::Rc;
use metal::{Buffer, MTLResourceOptions};

fn main() {
    println!("=== Single GEMM Profile ===\n");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/Users/jia/Desktop/infer_train/DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string());

    println!("Loading model from: {}", model_path);

    let ctx = InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Gpu,
    ).expect("Failed to load model");

    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");
    let device = gpu_ctx.device().clone();

    // Get Q projection weights
    let layer = &ctx.model.layers[0];
    let wq = layer.wq.as_ref().expect("No wq");
    let hidden_size = ctx.model.config.hidden_size;
    let q_out_size = wq.shape[1];

    println!("Model: hidden_size={}, q_out_size={}", hidden_size, q_out_size);
    println!("Weight shape: {:?}", wq.shape);

    // Get weight buffer
    let wq_buffer = gpu_ctx.get_or_create_buffer(wq);

    // Test different batch sizes
    let batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256];

    println!("\n| Batch | Time (ms) | GOPS |");
    println!("|-------|-----------|------|");

    // Warmup
    {
        let input_buffer = Rc::new(device.new_buffer(
            (hidden_size * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let output_buffer = Rc::new(device.new_buffer(
            (q_out_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        gpu_ctx.mul_mm_q5_k_encode(
            &encoder, &wq_buffer, &input_buffer, &output_buffer,
            q_out_size, 1, hidden_size,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    for batch in batch_sizes.iter() {
        // Create input buffer [batch, hidden_size] in F16
        let input_buffer = Rc::new(device.new_buffer(
            (batch * hidden_size * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        // Create output buffer [batch, q_out_size] in F32
        let output_buffer = Rc::new(device.new_buffer(
            (batch * q_out_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        // Run GEMM
        let start = std::time::Instant::now();

        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        gpu_ctx.mul_mm_q5_k_encode(
            &encoder, &wq_buffer, &input_buffer, &output_buffer,
            q_out_size, *batch, hidden_size,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let elapsed = start.elapsed();
        let flops = 2.0 * q_out_size as f64 * hidden_size as f64 * *batch as f64;
        let gops = flops / elapsed.as_secs_f64() / 1e9;

        println!("| {:5} | {:9.2} | {:6.1} |", batch, elapsed.as_secs_f64() * 1000.0, gops);
    }

    println!("\n=== Test Complete ===");
}
