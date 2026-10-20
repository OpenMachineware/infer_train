// Test multiple GEMM operations in single command buffer
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::inference::InferenceContext;
use std::rc::Rc;
use metal::{Buffer, MTLResourceOptions};

fn main() {
    println!("=== Multiple GEMM Test ===\n");

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/Users/jia/Desktop/infer_train/DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string());

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

    let wq_buffer = gpu_ctx.get_or_create_buffer(wq);

    let batch = 256;

    // Create input and output buffers
    let input_buffer = Rc::new(device.new_buffer(
        (batch * hidden_size * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    ));
    let output_buffer = Rc::new(device.new_buffer(
        (batch * q_out_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    ));

    // Warmup
    {
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        gpu_ctx.mul_mm_q5_k_encode(
            &encoder, &wq_buffer, &input_buffer, &output_buffer,
            q_out_size, batch, hidden_size,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Test 1: Single GEMM
    {
        let start = std::time::Instant::now();
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        gpu_ctx.mul_mm_q5_k_encode(
            &encoder, &wq_buffer, &input_buffer, &output_buffer,
            q_out_size, batch, hidden_size,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        println!("Single GEMM: {:.2} ms", start.elapsed().as_secs_f64() * 1000.0);
    }

    // Test 2: 7 GEMMs in single command buffer (same as one layer)
    {
        let start = std::time::Instant::now();
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        for _ in 0..7 {
            gpu_ctx.mul_mm_q5_k_encode(
                &encoder, &wq_buffer, &input_buffer, &output_buffer,
                q_out_size, batch, hidden_size,
            );
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let elapsed = start.elapsed();
        println!("7 GEMMs (one layer): {:.2} ms ({:.2} ms per GEMM)",
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / 7.0);
    }

    // Test 3: 196 GEMMs in single command buffer (all layers)
    {
        let start = std::time::Instant::now();
        let command_buffer = gpu_ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        for _ in 0..196 {
            gpu_ctx.mul_mm_q5_k_encode(
                &encoder, &wq_buffer, &input_buffer, &output_buffer,
                q_out_size, batch, hidden_size,
            );
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let elapsed = start.elapsed();
        println!("196 GEMMs (all layers): {:.2} ms ({:.2} ms per GEMM)",
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / 196.0);
    }

    println!("\n=== Test Complete ===");
}
