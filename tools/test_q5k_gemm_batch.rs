// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test Q5_K GEMM kernel with batch

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

        println!("\n=== Testing Q5_K GEMM with batch ===");
        println!("wq dtype={:?}, shape={:?}", wq.dtype, wq.shape);

        for batch in [1, 2].iter() {
            let batch = *batch;
            println!("\n--- Batch = {} ---", batch);

            let hidden_size = ctx.config.hidden_size;
            let q_out_size = wq.shape[1];

            // Create test input (F16)
            let input_data: Vec<u16> = (0..(batch * hidden_size))
                .map(|i| {
                    half::f16::from_f32(((i % hidden_size) as f32 % 10.0) * 0.1)
                        .to_bits()
                })
                .collect();

            let input_buffer = device.new_buffer_with_data(
                input_data.as_ptr() as *const std::ffi::c_void,
                (batch * hidden_size * 2) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            // Test Q projection
            let wq_buffer = gpu_ctx.get_or_create_buffer(wq);
            let q_output = device.new_buffer(
                (batch * q_out_size * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );

            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();

            gpu_ctx.mul_mm_q5_k_encode(
                encoder,
                &wq_buffer,
                &input_buffer,
                &q_output,
                q_out_size,
                batch,
                hidden_size,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            // Check Q output
            let q_data: Vec<f32> = unsafe {
                let ptr = q_output.contents() as *const f32;
                let mut data = vec![0.0f32; batch * q_out_size];
                std::ptr::copy_nonoverlapping(
                    ptr,
                    data.as_mut_ptr(),
                    batch * q_out_size,
                );
                data
            };
            let q_nan = q_data.iter().filter(|x| x.is_nan()).count();
            let q_valid: Vec<f32> =
                q_data.iter().filter(|x| x.is_finite()).cloned().collect();
            println!("Q: {} NaN out of {}", q_nan, q_data.len());
            if !q_valid.is_empty() {
                println!(
                    "  Valid range: [{:.6}, {:.6}]",
                    q_valid.iter().cloned().fold(f32::INFINITY, f32::min),
                    q_valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                );
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}
