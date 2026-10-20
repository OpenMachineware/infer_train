// Profile FFN GEMM operations
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::inference::InferenceContext;
use std::rc::Rc;
use metal::{Buffer, MTLResourceOptions};

fn main() {
    println!("=== FFN GEMM Profile ===\n");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/Users/jia/Desktop/infer_train/DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string());

    println!("Loading model from: {}", model_path);

    let ctx = InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Gpu,
    ).expect("Failed to load model");

    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");
    let device = gpu_ctx.device().clone();

    // Get FFN weights
    let layer = &ctx.model.layers[0];
    let w1 = layer.w1.as_ref().expect("No w1 (gate)");
    let w2 = layer.w2.as_ref().expect("No w2 (down)");
    let w3 = layer.w3.as_ref().expect("No w3 (up)");

    let hidden_size = ctx.model.config.hidden_size;
    let intermediate_size = w1.shape[1];

    println!("Hidden size: {}", hidden_size);
    println!("Intermediate size: {}", intermediate_size);
    println!("Gate weight shape: {:?}", w1.shape);
    println!("Up weight shape: {:?}", w3.shape);
    println!("Down weight shape: {:?}", w2.shape);

    // Get weight buffers
    let w1_buffer = gpu_ctx.get_or_create_buffer(w1);
    let w2_buffer = gpu_ctx.get_or_create_buffer(w2);
    let w3_buffer = gpu_ctx.get_or_create_buffer(w3);

    // Test different batch sizes
    let batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256];

    println!("\n=== Gate Projection ({}x{}) ===", intermediate_size, hidden_size);
    println!("| Batch | Time (ms) | GOPS |");
    println!("|-------|-----------|------|");

    // Warmup
    {
        let input_buffer = Rc::new(device.new_buffer(
            (hidden_size * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let output_buffer = Rc::new(device.new_buffer(
            (intermediate_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        gpu_ctx.mul_mm_q5_k_encode(
            &encoder, &w1_buffer, &input_buffer, &output_buffer,
            intermediate_size, 1, hidden_size,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    for batch in batch_sizes.iter() {
        let input_buffer = Rc::new(device.new_buffer(
            (batch * hidden_size * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let output_buffer = Rc::new(device.new_buffer(
            (batch * intermediate_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let start = std::time::Instant::now();
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        gpu_ctx.mul_mm_q5_k_encode(
            &encoder, &w1_buffer, &input_buffer, &output_buffer,
            intermediate_size, *batch, hidden_size,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let elapsed = start.elapsed();
        let flops = 2.0 * intermediate_size as f64 * hidden_size as f64 * *batch as f64;
        let gops = flops / elapsed.as_secs_f64() / 1e9;

        println!("| {:5} | {:9.2} | {:6.1} |", batch, elapsed.as_secs_f64() * 1000.0, gops);
    }

    println!("\n=== Down Projection ({}x{}) ===", hidden_size, intermediate_size);
    println!("| Batch | Time (ms) | GOPS |");
    println!("|-------|-----------|------|");

    for batch in batch_sizes.iter() {
        let input_buffer = Rc::new(device.new_buffer(
            (batch * intermediate_size * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        ));
        let output_buffer = Rc::new(device.new_buffer(
            (batch * hidden_size * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        ));

        let start = std::time::Instant::now();
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        gpu_ctx.mul_mm_q5_k_encode(
            &encoder, &w2_buffer, &input_buffer, &output_buffer,
            hidden_size, *batch, intermediate_size,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let elapsed = start.elapsed();
        let flops = 2.0 * hidden_size as f64 * intermediate_size as f64 * *batch as f64;
        let gops = flops / elapsed.as_secs_f64() / 1e9;

        println!("| {:5} | {:9.2} | {:6.1} |", batch, elapsed.as_secs_f64() * 1000.0, gops);
    }

    println!("\n=== Test Complete ===");
}
