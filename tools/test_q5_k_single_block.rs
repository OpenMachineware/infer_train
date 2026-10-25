// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test single Q5_K block dequantization vs GEMV

use std::env;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use infer_train::model::tensor::{Tensor, DataType, TensorData};
        use metal::*;

        let gpu_ctx = MetalContext::new().expect("Failed to create Metal context");
        let device = gpu_ctx.device();

        // Create a single Q5_K block with known values
        // Block layout: d[2] + dmin[2] + scales[12] + qh[32] + qs[128] = 176 bytes
        let mut block_data = vec![0u8; 176];

        // Set d and dmin
        let d = 0.01f32;
        let dmin = 0.001f32;
        let d_bits = half::f16::from_f32(d).to_bits();
        let dmin_bits = half::f16::from_f32(dmin).to_bits();
        block_data[0..2].copy_from_slice(&d_bits.to_le_bytes());
        block_data[2..4].copy_from_slice(&dmin_bits.to_le_bytes());

        // Set scales - use simple values: scales = 32, mins = 16
        // This makes (scale - 32) = 0 and (min - 32) = -16
        // So the formula becomes: d * 0 * q + dmin * (-16) = -0.016
        for i in 0..12 {
            block_data[4 + i] = 32 + i as u8; // scales[0-7] = 32-39
            // For packed format, we need to check the actual packing
        }

        // Set qs to all 8s (4-bit value 8)
        for i in 0..128 {
            block_data[48 + i] = 0x88; // low nibble = 8, high nibble = 8
        }

        // Set qh to 0 (no high bit contribution)
        for i in 0..32 {
            block_data[16 + i] = 0;
        }

        println!("=== Single Q5_K block test ===");
        println!("d = {}, dmin = {}", d, dmin);
        println!("Block size: {} bytes", block_data.len());

        // Create GPU buffer
        let block_buffer = device.new_buffer_with_data(
            block_data.as_ptr() as *const std::ffi::c_void,
            176u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Test GEMV: compute dot product with all-1s input
        let input: Vec<f32> = vec![1.0; 256];
        let input_buffer = device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            1024u64,
            MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = device.new_buffer(
            4u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create a tensor wrapper
        let wq_tensor = Tensor {
            name: "test_wq".to_string(),
            shape: vec![1, 256], // 1 row, 256 columns
            dtype: DataType::Q5_K,
            data: TensorData::Owned(block_data.clone()),
            #[cfg(all(target_arch = "aarch64", feature = "metal"))]
            gpu_buffer: None,
        };

        let wq_gpu_buffer = gpu_ctx.get_or_create_buffer(&wq_tensor);

        let cb = gpu_ctx.queue().new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        gpu_ctx.mv_encode(
            &encoder,
            &wq_gpu_buffer,
            &input_buffer,
            &output_buffer,
            1,   // M (output size)
            256, // K (input size)
            DataType::Q5_K,
        );
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let gemv_result: f32 = unsafe {
            let ptr = output_buffer.contents() as *const f32;
            *ptr
        };

        println!("\nGEMV result (should be sum of all dequantized values): {}", gemv_result);

        // Expected: for each position, value = d * (scale-32) * q + dmin * (min-32)
        // With scale=32, min=16, q=8: d * 0 * 8 + dmin * (-16) = -0.016
        // Total for 256 elements: 256 * (-0.016) = -4.096
        println!("Expected: 256 * (d * (32-32) * 8 + dmin * (16-32)) = {}", 256.0 * (d * 0.0 * 8.0 + dmin * (16.0 - 32.0)));
    }
}
