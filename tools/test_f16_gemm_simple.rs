// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test simple F16 GEMM to verify kernel logic (without quantization complexity)

use std::env;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let args: Vec<String> = env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: {} <model.gguf>", args[0]);
            std::process::exit(1);
        }

        let model_path = &args[1];
        println!("Model: {}", model_path);

        // Load model just to initialize Metal context
        let ctx = infer_train::inference::InferenceContext::from_gguf(
            model_path,
            infer_train::inference::Device::Cpu,
        )
        .expect("Failed to load model");

        let gpu_ctx =
            infer_train::quant::vec_dot::gpu::metal::MetalContext::new()
                .expect("Failed to create Metal context");
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        let hidden_size = ctx.config.hidden_size;

        println!("\n=== Testing simple F16 GEMM ===");
        println!("Testing with small matrix: M=64, N=2, K={}", hidden_size);

        // Create F16 weights [M, K] = [64, hidden_size]
        // Fill with small values to avoid overflow
        let weight_data: Vec<u16> = (0..(64 * hidden_size))
            .map(|i| half::f16::from_f32(0.01).to_bits())
            .collect();

        let weight_buffer = device.new_buffer_with_data(
            weight_data.as_ptr() as *const std::ffi::c_void,
            (64 * hidden_size * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create F16 input [N, K] = [2, hidden_size]
        // Batch 0: all 1.0
        // Batch 1: all 2.0
        let mut input_data = vec![0u16; 2 * hidden_size];
        for i in 0..hidden_size {
            input_data[i] = half::f16::from_f32(1.0).to_bits();
            input_data[hidden_size + i] = half::f16::from_f32(2.0).to_bits();
        }

        let input_buffer = device.new_buffer_with_data(
            input_data.as_ptr() as *const std::ffi::c_void,
            (2 * hidden_size * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Output [M, N] = [64, 2] as F32
        let output_buffer = device.new_buffer(
            (64 * 2 * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Call F16 GEMM
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        gpu_ctx.mul_mm_f16_encode(
            &encoder,
            &weight_buffer,
            &input_buffer,
            &output_buffer,
            64,          // M
            2,           // N
            hidden_size, // K
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let output_data: Vec<f32> = unsafe {
            let ptr = output_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; 128];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), 128);
            data
        };

        // Expected: Y = W @ X
        // With W all 0.01 and X all 1.0 (batch 0) or 2.0 (batch 1):
        // Y[i, 0] = sum(0.01 * 1.0) * K = 0.01 * K
        // Y[i, 1] = sum(0.01 * 2.0) * K = 0.02 * K

        println!("\nExpected values:");
        println!(
            "  Batch 0: each element = 0.01 * {} = {}",
            hidden_size,
            0.01 * hidden_size as f32
        );
        println!(
            "  Batch 1: each element = 0.02 * {} = {}",
            hidden_size,
            0.02 * hidden_size as f32
        );

        println!("\nActual values (first 10 elements):");
        for i in 0..10.min(64) {
            println!("  Y[{}, 0] = {:.4} (batch 0)", i, output_data[i]);
            println!("  Y[{}, 1] = {:.4} (batch 1)", i, output_data[64 + i]);
        }

        let batch0: Vec<f32> = output_data[0..64].to_vec();
        let batch1: Vec<f32> = output_data[64..128].to_vec();

        let b0_nan = batch0.iter().filter(|x| x.is_nan()).count();
        let b1_nan = batch1.iter().filter(|x| x.is_nan()).count();

        let b0_valid: Vec<f32> =
            batch0.iter().filter(|x| x.is_finite()).cloned().collect();
        let b1_valid: Vec<f32> =
            batch1.iter().filter(|x| x.is_finite()).cloned().collect();

        println!("\nNaN count: batch0={}, batch1={}", b0_nan, b1_nan);
        if !b0_valid.is_empty() {
            println!(
                "Batch 0 range: [{:.4}, {:.4}]",
                b0_valid.iter().cloned().fold(f32::INFINITY, f32::min),
                b0_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }
        if !b1_valid.is_empty() {
            println!(
                "Batch 1 range: [{:.4}, {:.4}]",
                b1_valid.iter().cloned().fold(f32::INFINITY, f32::min),
                b1_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }

        // Check ratio
        if !b0_valid.is_empty() && !b1_valid.is_empty() {
            let ratio = b1_valid[0] / b0_valid[0];
            println!("\nBatch 1 / Batch 0 ratio: {:.4} (expected 2.0)", ratio);
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}
