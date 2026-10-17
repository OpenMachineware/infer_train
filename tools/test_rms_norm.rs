// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test RMS Norm GPU kernel independently

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    println!("Loading model: {}", model_path);
    let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        if let Some(ref gpu_ctx) = ctx.gpu_ctx {
            let model = &ctx.model;
            let device = gpu_ctx.device();
            let queue = gpu_ctx.queue();

            // Create test input (non-zero values)
            let hidden_size = model.config.hidden_size;
            let input_data: Vec<f32> =
                (0..hidden_size).map(|i| (i as f32) * 0.01).collect();

            println!("\n=== Test RMS Norm ===");
            println!(
                "Input: {} values, first 5: {:?}",
                hidden_size,
                &input_data[..5]
            );

            // Upload to GPU
            use metal::MTLResourceOptions;
            let input_buffer = std::rc::Rc::new(device.new_buffer_with_data(
                input_data.as_ptr() as *const std::ffi::c_void,
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            // Get norm weight
            let norm = model.norm.as_ref().expect("No final norm");
            let norm_buffer = gpu_ctx.get_or_create_buffer(norm);

            // Create output buffer
            let output_buffer = std::rc::Rc::new(device.new_buffer(
                (hidden_size * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ));

            // Encode and run
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            gpu_ctx.rms_norm_encode(
                encoder,
                &input_buffer,
                &norm_buffer,
                &output_buffer,
                1,
                hidden_size,
                1e-5,
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read output
            let mut output_data = vec![0.0f32; hidden_size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    output_buffer.contents() as *const f32,
                    output_data.as_mut_ptr(),
                    hidden_size,
                );
            }

            let num_zero = output_data.iter().filter(|&&v| v == 0.0).count();
            let min = output_data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max =
                output_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            println!(
                "Output: {} NaN, {} zero, range [{:.4}, {:.4}]",
                output_data.iter().filter(|&&v| v.is_nan()).count(),
                num_zero,
                min,
                max
            );
            println!("First 5: {:?}", &output_data[..5]);

            if num_zero > 0 {
                println!("WARNING: RMS Norm produced zeros!");
            }
        }
    }
}
