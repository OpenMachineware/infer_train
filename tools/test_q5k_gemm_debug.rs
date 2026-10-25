// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test Q5_K GEMM with small M dimension to match F16 test

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

        // Load model just to get weight tensor
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

        println!("\n=== Testing Q5_K GEMM with M=64 ===");
        println!("M=64, N=2, K={}", hidden_size);

        // Get a small subset of the wq weights (first 64 rows)
        let layer = ctx.model.layers.first().expect("No layers");
        let wq = layer.wq.as_ref().expect("No wq");

        // Create a small Q5_K weight tensor with 64 rows
        // This requires extracting 64 rows from the full weights
        // For simplicity, let's just use the first layer's wq weights
        // and test with full M dimension to see if the issue is consistent

        println!(
            "\nUsing full wq tensor: shape={:?}, dtype={:?}",
            wq.shape, wq.dtype
        );

        let m = wq.shape[1]; // Full M dimension

        // Create F16 input [N, K] = [2, hidden_size]
        let mut input_data = vec![0u16; 2 * hidden_size];
        for i in 0..hidden_size {
            input_data[i] = half::f16::from_f32(0.1).to_bits(); // Smaller values
            input_data[hidden_size + i] = half::f16::from_f32(0.2).to_bits();
        }

        let input_buffer = device.new_buffer_with_data(
            input_data.as_ptr() as *const std::ffi::c_void,
            (2 * hidden_size * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let wq_buffer = gpu_ctx.get_or_create_buffer(wq);
        let output_buffer = device.new_buffer(
            (m * 2 * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Q5_K GEMM
        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        gpu_ctx.mul_mm_q5_k_encode(
            &encoder,
            &wq_buffer,
            &input_buffer,
            &output_buffer,
            m,
            2,
            hidden_size,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let output_data: Vec<f32> = unsafe {
            let ptr = output_buffer.contents() as *const f32;
            let mut data = vec![0.0f32; m * 2];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), m * 2);
            data
        };

        let batch0 = &output_data[0..m];
        let batch1 = &output_data[m..2 * m];

        let b0_nan = batch0.iter().filter(|x| x.is_nan()).count();
        let b1_nan = batch1.iter().filter(|x| x.is_nan()).count();

        let b0_valid: Vec<f32> =
            batch0.iter().filter(|x| x.is_finite()).cloned().collect();
        let b1_valid: Vec<f32> =
            batch1.iter().filter(|x| x.is_finite()).cloned().collect();

        println!("\nResults:");
        println!("  NaN count: batch0={}, batch1={}", b0_nan, b1_nan);

        if !b0_valid.is_empty() {
            println!(
                "  Batch 0 range: [{:.6}, {:.6}]",
                b0_valid.iter().cloned().fold(f32::INFINITY, f32::min),
                b0_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }
        if !b1_valid.is_empty() {
            println!(
                "  Batch 1 range: [{:.6}, {:.6}]",
                b1_valid.iter().cloned().fold(f32::INFINITY, f32::min),
                b1_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }

        // Check if batch 1 is approximately 2x batch 0
        if !b0_valid.is_empty()
            && !b1_valid.is_empty()
            && b0_valid[0].abs() > 0.01
        {
            let ratios: Vec<f32> = b0_valid
                .iter()
                .zip(b1_valid.iter())
                .take(10)
                .map(|(a, b)| b / a)
                .collect();
            println!("  Batch 1 / Batch 0 ratios (first 10): {:?}", ratios);
        }

        // Compare with GEMV for batch 0
        println!("\nComparing with GEMV for batch 0:");
        // GEMV expects F32 input!
        let single_input_f32: Vec<f32> =
            (0..hidden_size).map(|_| 0.1).collect();
        let single_input_buffer = device.new_buffer_with_data(
            single_input_f32.as_ptr() as *const std::ffi::c_void,
            (hidden_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let gemv_output = device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let cb = queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        gpu_ctx.mv_encode(
            encoder,
            &wq_buffer,
            &single_input_buffer,
            &gemv_output,
            m,
            hidden_size,
            wq.dtype,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let gemv_data: Vec<f32> = unsafe {
            let ptr = gemv_output.contents() as *const f32;
            let mut data = vec![0.0f32; m];
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), m);
            data
        };

        let gemv_nan = gemv_data.iter().filter(|x| x.is_nan()).count();
        let gemv_valid: Vec<f32> =
            gemv_data.iter().filter(|x| x.is_finite()).cloned().collect();

        println!("  GEMV NaN count: {}", gemv_nan);
        if !gemv_valid.is_empty() {
            println!(
                "  GEMV range: [{:.6}, {:.6}]",
                gemv_valid.iter().cloned().fold(f32::INFINITY, f32::min),
                gemv_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }

        // Compare GEMM batch 0 vs GEMV
        if !b0_valid.is_empty() && !gemv_valid.is_empty() {
            let mut max_rel_err = 0.0f32;
            let mut err_count = 0;

            for i in 0..m.min(100) {
                if batch0[i].is_finite()
                    && gemv_data[i].is_finite()
                    && gemv_data[i].abs() > 0.01
                {
                    let rel_err =
                        ((batch0[i] - gemv_data[i]) / gemv_data[i]).abs();
                    if rel_err > max_rel_err {
                        max_rel_err = rel_err;
                    }
                    if rel_err > 0.01 {
                        err_count += 1;
                        if err_count <= 5 {
                            println!(
                                "    idx {}: GEMM={:.6}, GEMV={:.6}, err={:.4}%",
                                i,
                                batch0[i],
                                gemv_data[i],
                                rel_err * 100.0
                            );
                        }
                    }
                }
            }

            println!("  Max relative error: {:.4}%", max_rel_err * 100.0);
            println!("  Values with >1% error: {}", err_count);
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}
