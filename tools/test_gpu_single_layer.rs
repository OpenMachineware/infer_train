// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test a single layer on GPU to debug zero output issue

use infer_train::inference::{Device, InferenceContext};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::MTLResourceOptions;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model: {}", model_path);

    // Load model
    let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    println!("Model loaded on GPU");
    println!("Layers: {}", ctx.layer_count());

    // Get model data
    let model = &ctx.model;
    let config = &ctx.config;
    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");

    let device = gpu_ctx.device();
    let queue = gpu_ctx.queue();

    // Test Layer 0: RMS Norm + Q projection
    let layer = &model.layers[0];
    let hidden_size = config.hidden_size;

    println!("\n=== Test Layer 0 ===");
    println!("Hidden size: {}", hidden_size);

    // Create input embedding (non-zero test data)
    let hidden: Vec<f32> =
        (0..hidden_size).map(|i| (i as f32) * 0.001).collect();
    println!(
        "Input hidden: {} values, range [{}, {}]",
        hidden.len(),
        hidden.iter().cloned().fold(f32::INFINITY, f32::min),
        hidden.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // Create buffers
    let hidden_buffer = std::rc::Rc::new(device.new_buffer_with_data(
        hidden.as_ptr() as *const std::ffi::c_void,
        (hidden_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    ));

    // Test RMS Norm
    println!("\n--- Testing RMS Norm ---");
    let attention_norm =
        layer.attention_norm.as_ref().expect("No attention norm");
    println!(
        "Attention norm: dtype={:?}, shape={:?}",
        attention_norm.dtype, attention_norm.shape
    );

    let normed_buffer = std::rc::Rc::new(device.new_buffer(
        (hidden_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    ));

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    let norm_weight_buffer = gpu_ctx.get_or_create_buffer(attention_norm);
    gpu_ctx.rms_norm_encode(
        encoder,
        &hidden_buffer,
        &norm_weight_buffer,
        &normed_buffer,
        1,
        hidden_size,
        1e-5,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back
    let mut normed = vec![0.0f32; hidden_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            normed_buffer.contents() as *const f32,
            normed.as_mut_ptr(),
            hidden_size,
        );
    }

    let num_zero = normed.iter().filter(|&&v| v == 0.0).count();
    let num_nan = normed.iter().filter(|&&v| v.is_nan()).count();
    println!(
        "RMS Norm output: {} values, {} zero, {} nan, range [{}, {}]",
        normed.len(),
        num_zero,
        num_nan,
        normed.iter().cloned().fold(f32::INFINITY, f32::min),
        normed.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("First 5: {:?}", &normed[..5]);

    // Test Q projection
    println!("\n--- Testing Q Projection ---");
    let wq = layer.wq.as_ref().expect("No wq");
    println!("WQ: dtype={:?}, shape={:?}", wq.dtype, wq.shape);
    let q_out_size = wq.shape[1];

    let q_buffer = std::rc::Rc::new(device.new_buffer(
        (q_out_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    ));

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    let result = model.gemv_encode(
        encoder,
        &normed_buffer,
        wq,
        q_out_size,
        gpu_ctx,
        &q_buffer,
    );

    println!("GEMV encode result: {:?}", result.is_some());

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back
    let mut q = vec![0.0f32; q_out_size];
    unsafe {
        std::ptr::copy_nonoverlapping(
            q_buffer.contents() as *const f32,
            q.as_mut_ptr(),
            q_out_size,
        );
    }

    let num_zero = q.iter().filter(|&&v| v == 0.0).count();
    let num_nan = q.iter().filter(|&&v| v.is_nan()).count();
    println!(
        "Q output: {} values, {} zero, {} nan, range [{}, {}]",
        q.len(),
        num_zero,
        num_nan,
        q.iter().cloned().fold(f32::INFINITY, f32::min),
        q.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("First 5: {:?}", &q[..5]);

    // Compare with CPU
    println!("\n--- Comparing with CPU ---");
    let cpu_q = model.gemv(&normed, wq, q_out_size).expect("CPU gemv failed");
    let num_zero = cpu_q.iter().filter(|&&v| v == 0.0).count();
    let num_nan = cpu_q.iter().filter(|&&v| v.is_nan()).count();
    println!(
        "CPU Q: {} values, {} zero, {} nan, range [{}, {}]",
        cpu_q.len(),
        num_zero,
        num_nan,
        cpu_q.iter().cloned().fold(f32::INFINITY, f32::min),
        cpu_q.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!("First 5: {:?}", &cpu_q[..5]);

    // Calculate relative error
    let mut max_rel_error = 0.0f32;
    let mut avg_rel_error = 0.0f32;
    let mut count = 0;
    for i in 0..q_out_size {
        if cpu_q[i].abs() > 1e-6 {
            let rel_error = (q[i] - cpu_q[i]).abs() / cpu_q[i].abs();
            max_rel_error = max_rel_error.max(rel_error);
            avg_rel_error += rel_error;
            count += 1;
        }
    }
    if count > 0 {
        avg_rel_error /= count as f32;
    }
    println!("Relative error: max={}, avg={}", max_rel_error, avg_rel_error);

    println!("\n=== Done ===");
}
