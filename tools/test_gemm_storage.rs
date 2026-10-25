// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test GEMM storage pattern by examining specific output values

use std::env;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let args: Vec<String> = env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: {} <model.gguf>", args[0]);
            std::process::exit(1);
        }

        let model_path = &args[1];
        println!("Model: {}", model_path);

        let gpu_ctx =
            MetalContext::new().expect("Failed to create Metal context");
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        // Load model
        let ctx = infer_train::inference::InferenceContext::from_gguf(
            model_path,
            infer_train::inference::Device::Cpu,
        )
        .expect("Failed to load model");

        let layer = ctx.model.layers.first().expect("No layers");
        let wq = layer.wq.as_ref().expect("No wq");

        let hidden_size = ctx.config.hidden_size;
        let q_out_size = wq.shape[1];

        println!("\n=== Testing GEMM output pattern ===");
        println!("wq shape: {:?}", wq.shape);
        println!("hidden_size (K): {}", hidden_size);
        println!("q_out_size (M): {}", q_out_size);

        // Create test input with distinct patterns for each batch
        // Batch 0: all 1.0
        // Batch 1: all 2.0
        let mut input_data = vec![0u16; 2 * hidden_size];
        for i in 0..hidden_size {
            input_data[i] = half::f16::from_f32(1.0).to_bits(); // Batch 0
            input_data[hidden_size + i] = half::f16::from_f32(2.0).to_bits(); // Batch 1
        }

        let input_buffer = device.new_buffer_with_data(
            input_data.as_ptr() as *const std::ffi::c_void,
            (2 * hidden_size * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let wq_buffer = gpu_ctx.get_or_create_buffer(wq);
        let gemm_output = device.new_buffer(
            (2 * q_out_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // GEMM: batch=2
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        gpu_ctx.mul_mm_q5_k_encode(
            encoder,
            &wq_buffer,
            &input_buffer,
            &gemm_output,
            q_out_size,
            2, // batch
            hidden_size,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let gemm_data: Vec<f32> = unsafe {
            let ptr = gemm_output.contents() as *const f32;
            let mut data = vec![0.0f32; 2 * q_out_size];
            std::ptr::copy_nonoverlapping(
                ptr,
                data.as_mut_ptr(),
                2 * q_out_size,
            );
            data
        };

        // Check the relationship between batch 0 and batch 1
        // If Y is column-major [M, N], then:
        // - Batch 0 (col 0): elements 0 to M-1
        // - Batch 1 (col 1): elements M to 2M-1
        // If the kernel is correct, batch 1 values should be ~2x batch 0

        println!("\nChecking if batch 1 = 2x batch 0:");
        let batch0 = &gemm_data[0..q_out_size];
        let batch1 = &gemm_data[q_out_size..2 * q_out_size];

        let mut matches = 0;
        let mut mismatches = 0;
        let mut batch0_nans = 0;
        let mut batch1_nans = 0;

        for i in 0..q_out_size.min(100) {
            if batch0[i].is_nan() {
                batch0_nans += 1;
                continue;
            }
            if batch1[i].is_nan() {
                batch1_nans += 1;
                continue;
            }

            let ratio = if batch0[i].abs() > 0.001 {
                batch1[i] / batch0[i]
            } else {
                0.0
            };

            if (ratio - 2.0).abs() < 0.1 {
                matches += 1;
            } else {
                mismatches += 1;
                if mismatches <= 10 {
                    println!(
                        "  idx {}: batch0={:.4}, batch1={:.4}, ratio={:.4}",
                        i, batch0[i], batch1[i], ratio
                    );
                }
            }
        }

        println!("\nResults:");
        println!("  Batch 0 NaNs: {}", batch0_nans);
        println!("  Batch 1 NaNs: {}", batch1_nans);
        println!("  Matches (ratio ≈ 2.0): {}", matches);
        println!("  Mismatches: {}", mismatches);

        if mismatches == 0 && matches > 0 {
            println!("\n✓ Batch 1 = 2x Batch 0 (correct relationship)");
        } else if matches == 0 {
            println!("\n✗ No relationship found between batches");
        }

        // Also check value ranges
        let b0_valid: Vec<f32> =
            batch0.iter().filter(|x| x.is_finite()).cloned().collect();
        let b1_valid: Vec<f32> =
            batch1.iter().filter(|x| x.is_finite()).cloned().collect();

        if !b0_valid.is_empty() {
            println!(
                "\nBatch 0 range: [{:.6}, {:.6}]",
                b0_valid.iter().cloned().fold(f32::INFINITY, f32::min),
                b0_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }
        if !b1_valid.is_empty() {
            println!(
                "Batch 1 range: [{:.6}, {:.6}]",
                b1_valid.iter().cloned().fold(f32::INFINITY, f32::min),
                b1_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}
