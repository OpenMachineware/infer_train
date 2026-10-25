// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test GEMV (decode) vs GEMM (prefill) outputs to verify batched GEMM correctness

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

        println!("\n=== Comparing GEMV (batch=1) vs GEMM (batch=2) ===");
        println!("wq dtype={:?}, shape={:?}", wq.dtype, wq.shape);

        let hidden_size = ctx.config.hidden_size;
        let q_out_size = wq.shape[1];

        // Create test input (F16)
        // For batch=2: [input0, input1]
        // For batch=1 twice: should get same results
        let input_data: Vec<u16> = (0..(2 * hidden_size))
            .map(|i| {
                half::f16::from_f32(((i % hidden_size) as f32 % 10.0) * 0.1)
                    .to_bits()
            })
            .collect();

        // === GEMM: Process both inputs together ===
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

        // === GEMV: Process each input separately ===
        let mut gemv_results = Vec::new();

        for batch_idx in 0..2 {
            let single_input: Vec<u16> = input_data
                [batch_idx * hidden_size..(batch_idx + 1) * hidden_size]
                .to_vec();
            let single_input_buffer = device.new_buffer_with_data(
                single_input.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let gemv_output = device.new_buffer(
                (q_out_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            gpu_ctx.mv_encode(
                encoder,
                &wq_buffer,
                &single_input_buffer,
                &gemv_output,
                q_out_size,
                hidden_size,
                wq.dtype,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            let gemv_data: Vec<f32> = unsafe {
                let ptr = gemv_output.contents() as *const f32;
                let mut data = vec![0.0f32; q_out_size];
                std::ptr::copy_nonoverlapping(
                    ptr,
                    data.as_mut_ptr(),
                    q_out_size,
                );
                data
            };

            gemv_results.push(gemv_data);
        }

        // Compare GEMM batch 0 vs GEMV 0
        println!("\n--- Comparing batch 0: GEMM vs GEMV ---");
        compare_outputs(&gemm_data[0..q_out_size], &gemv_results[0]);

        // Compare GEMM batch 1 vs GEMV 1
        println!("\n--- Comparing batch 1: GEMM vs GEMV ---");
        compare_outputs(
            &gemm_data[q_out_size..2 * q_out_size],
            &gemv_results[1],
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}

fn compare_outputs(gemm: &[f32], gemv: &[f32]) {
    let n = gemm.len();
    let nan_gemm = gemm.iter().filter(|x| x.is_nan()).count();
    let nan_gemv = gemv.iter().filter(|x| x.is_nan()).count();

    println!("NaN: GEMM={}, GEMV={}", nan_gemm, nan_gemv);

    if nan_gemv > 0 {
        println!("GEMV has NaN, cannot compare");
        return;
    }

    let gemv_valid: Vec<f32> =
        gemv.iter().filter(|x| x.is_finite()).cloned().collect();
    if !gemv_valid.is_empty() {
        println!(
            "GEMV valid range: [{:.6}, {:.6}]",
            gemv_valid.iter().cloned().fold(f32::INFINITY, f32::min),
            gemv_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
    }

    let gemm_valid: Vec<f32> =
        gemm.iter().filter(|x| x.is_finite()).cloned().collect();
    if !gemm_valid.is_empty() {
        println!(
            "GEMM valid range: [{:.6}, {:.6}]",
            gemm_valid.iter().cloned().fold(f32::INFINITY, f32::min),
            gemm_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
    }

    // Compute relative error for non-NaN values
    let mut max_rel_err = 0.0f32;
    let mut max_abs_err = 0.0f32;
    let mut err_count = 0;

    for i in 0..n {
        if gemm[i].is_finite() && gemv[i].is_finite() {
            let abs_err = (gemm[i] - gemv[i]).abs();
            let rel_err = if gemv[i].abs() > 1e-6 {
                abs_err / gemv[i].abs()
            } else {
                abs_err
            };

            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
            if abs_err > max_abs_err {
                max_abs_err = abs_err;
            }

            if rel_err > 0.01 {
                err_count += 1;
                if err_count <= 5 {
                    println!(
                        "  idx {}: GEMM={:.6}, GEMV={:.6}, rel_err={:.4}%",
                        i,
                        gemm[i],
                        gemv[i],
                        rel_err * 100.0
                    );
                }
            }
        }
    }

    println!("Max relative error: {:.6}%", max_rel_err * 100.0);
    println!("Max absolute error: {:.6}", max_abs_err);
    println!("Values with >1% error: {}", err_count);

    if err_count == 0 && max_rel_err < 0.001 {
        println!("✓ GEMM matches GEMV!");
    } else if err_count == 0 && max_rel_err < 0.01 {
        println!("✓ GEMM matches GEMV (within 1%)");
    } else {
        println!("✗ GEMM does NOT match GEMV");
    }
}
