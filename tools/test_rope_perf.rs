// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile decode execution time breakdown

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::{Device, InferenceContext};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let model_path = &args[1];
        println!("=== Decode Time Breakdown ===\n");

        let gpu_ctx =
            infer_train::quant::vec_dot::gpu::metal::MetalContext::new()
                .unwrap();

        // Test RoPE performance
        println!("Testing RoPE kernel...\n");
        test_rope_perf(&gpu_ctx);

        // Test RMS Norm performance
        println!("Testing RMS Norm kernel...\n");
        test_rms_norm_perf(&gpu_ctx);

        // Summary
        println!("=== Summary ===\n");
        println!("Known times:");
        println!("  Flash Attention: 341µs/layer (64% of decode)");
        println!("  Output projection: 1.78ms");
        println!();
        println!("Need to test:");
        println!("  FusedQKV: ? µs/layer");
        println!("  FusedGateUp: ? µs/layer");
        println!("  FusedSwiGLU: ? µs/layer");
        println!("  RoPE: ? µs/layer");
        println!("  RMS Norm: ? µs/layer");
        println!("  FusedAddNorm: ? µs/layer");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn test_rope_perf(
    gpu_ctx: &infer_train::quant::vec_dot::gpu::metal::MetalContext,
) {
    use std::rc::Rc;

    let device = gpu_ctx.device();
    let queue = gpu_ctx.queue();

    // DeepSeek config
    let n_heads = 12;
    let head_dim = 128;
    let seq_len = 1;
    let n_rot = head_dim / 2;

    let q_size = n_heads * head_dim;
    let q_data: Vec<f32> = vec![0.1; q_size];

    let q_buffer = Rc::new(device.new_buffer_with_data(
        q_data.as_ptr() as *const std::ffi::c_void,
        (q_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    let output_buffer = Rc::new(device.new_buffer(
        (q_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    // Warmup
    for _ in 0..5 {
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        gpu_ctx.rope_f32_encode(
            encoder,
            &q_buffer,
            &output_buffer,
            n_heads,
            head_dim,
            seq_len,
            n_rot,
            0, // pos
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Measure
    let iterations = 50;
    let mut times = Vec::new();
    for pos in 0..iterations {
        let start = std::time::Instant::now();
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        gpu_ctx.rope_f32_encode(
            encoder,
            &q_buffer,
            &output_buffer,
            n_heads,
            head_dim,
            seq_len,
            n_rot,
            pos,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }

    let avg: std::time::Duration =
        times.iter().sum::<std::time::Duration>() / iterations as u32;
    println!("RoPE: {:?}", avg);
}

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn test_rms_norm_perf(
    gpu_ctx: &infer_train::quant::vec_dot::gpu::metal::MetalContext,
) {
    use std::rc::Rc;

    let device = gpu_ctx.device();
    let queue = gpu_ctx.queue();

    let hidden_size = 1536;
    let data: Vec<f32> = vec![0.1; hidden_size];
    let weight: Vec<f32> = vec![1.0; hidden_size];

    let input_buffer = Rc::new(device.new_buffer_with_data(
        data.as_ptr() as *const std::ffi::c_void,
        (hidden_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    let weight_buffer = Rc::new(device.new_buffer_with_data(
        weight.as_ptr() as *const std::ffi::c_void,
        (hidden_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    let output_buffer = Rc::new(device.new_buffer(
        (hidden_size * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    ));

    // Warmup
    for _ in 0..5 {
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        gpu_ctx.rms_norm_encode(
            encoder,
            &input_buffer,
            &weight_buffer,
            &output_buffer,
            1,
            hidden_size,
            1e-5,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Measure
    let iterations = 50;
    let mut times = Vec::new();
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        gpu_ctx.rms_norm_encode(
            encoder,
            &input_buffer,
            &weight_buffer,
            &output_buffer,
            1,
            hidden_size,
            1e-5,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }

    let avg: std::time::Duration =
        times.iter().sum::<std::time::Duration>() / iterations as u32;
    println!("RMS Norm: {:?}\n", avg);
}
