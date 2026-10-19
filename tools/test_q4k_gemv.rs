// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Q4_K GEMV kernel in isolation

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use metal::MTLSize;

        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        // Test with Layer 8 weights (Q4_K)
        let layer = &ctx.model.layers[8];
        println!("=== Testing Layer 8 Q4_K GEMV ===");

        let wq = layer.wq.as_ref().unwrap();
        println!("wq: dtype={:?}, shape={:?}", wq.dtype, wq.shape);

        // Create test input (1024 floats)
        let input_size = 1024;
        let input_data: Vec<f32> = vec![0.1; input_size];
        let input_buffer = device.new_buffer_with_data(
            input_data.as_ptr() as *const std::ffi::c_void,
            (input_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create output buffer
        let output_size = wq.shape[1]; // 2048
        let output_buffer = device.new_buffer(
            (output_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Get weight buffer
        let weight_buffer = gpu_ctx.get_or_create_buffer(wq);
        println!("Weight buffer size: {} bytes", weight_buffer.length());

        // Run GEMV
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();

        let success = gpu_ctx.mv_encode(
            enc,
            &weight_buffer,
            &input_buffer,
            &output_buffer,
            output_size,
            input_size,
            wq.dtype,
        );

        if !success {
            println!("mv_encode returned false!");
            return;
        }

        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // Check output
        let output_ptr = output_buffer.contents() as *const f32;
        let mut output = vec![0.0f32; output_size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_ptr,
                output.as_mut_ptr(),
                output_size,
            );
        }

        let nan_count = output.iter().filter(|v| v.is_nan()).count();
        let zero_count = output.iter().filter(|v| **v == 0.0).count();
        let non_zero_count = output_size - zero_count;

        println!("Output stats:");
        println!("  Total: {}", output_size);
        println!("  NaN: {}", nan_count);
        println!("  Zero: {}", zero_count);
        println!("  Non-zero: {}", non_zero_count);
        println!("  First 10: {:?}", &output[..10]);

        if non_zero_count == 0 {
            println!("\n*** ERROR: All outputs are zero! ***");
        } else if nan_count > 0 {
            println!("\n*** ERROR: NaN values detected! ***");
        } else {
            println!("\n*** SUCCESS: Q4_K GEMV works correctly! ***");
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
