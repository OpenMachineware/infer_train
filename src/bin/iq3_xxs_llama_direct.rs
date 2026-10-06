// Verify IQ3_XXS GEMM with llama.cpp generated data
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use std::ffi::c_void;

const QK_K: usize = 256;

fn main() {
    println!("=== IQ3_XXS GEMM Verification with llama.cpp Data ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Test size
    let m = 1usize;
    let k = 256usize;
    let n = 1usize;

    // Test data (seed=42, same as llama.cpp dump)
    let x: Vec<f32> =
        (0..m * k).map(|i| ((i as f32 * 0.1 + 42.0) % 2.0) - 1.0).collect();
    let input_f16: Vec<u16> =
        x.iter().map(|&v| f16::from_f32(v).to_bits()).collect();

    // llama.cpp generated block
    let block = BlockIQ3XXS {
        d: 0x1999,
        qs: [
            0xa8, 0x1e, 0x00, 0x76, 0xc0, 0xa8, 0x1e, 0x00, 0x76, 0xc0, 0xa8,
            0x1e, 0x00, 0x76, 0xc0, 0xa8, 0x1e, 0x00, 0x76, 0xc0, 0xa8, 0x1e,
            0x00, 0x76, 0xc0, 0xa8, 0x1e, 0x00, 0x76, 0xc0, 0xa8, 0x1e, 0x00,
            0x76, 0xc0, 0xa8, 0x1e, 0x00, 0x76, 0xc0, 0xa8, 0x1e, 0x00, 0x76,
            0xc0, 0xa8, 0x1e, 0x00, 0x76, 0xc0, 0xa8, 0x1e, 0x00, 0x76, 0xc0,
            0xa8, 0x1e, 0x00, 0x76, 0xc0, 0xa8, 0x1e, 0x00, 0x76, 0xff, 0x01,
            0xfc, 0xf7, 0x80, 0xff, 0x00, 0xfe, 0x3f, 0xc0, 0x7f, 0xf0, 0xf0,
            0x1f, 0xe0, 0xff, 0x03, 0xf8, 0x0f, 0xf0, 0xff, 0x01, 0xfc, 0xf7,
            0x80, 0xff, 0x00, 0xfe, 0x3f, 0xc0, 0x7f, 0xf0,
        ],
    };

    let weights: Vec<BlockIQ3XXS> = vec![block];

    // Load Metal libraries
    let my_lib_path =
        std::path::Path::new("shaders/mul_mm_standalone.metallib");
    let my_lib = device
        .new_library_with_file(my_lib_path)
        .expect("Failed to load standalone library");

    let llama_lib_path =
        std::path::Path::new("shaders/mul_mm_template.metallib");
    let llama_lib = device
        .new_library_with_file(llama_lib_path)
        .expect("Failed to load template library");

    // Create buffers
    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * std::mem::size_of::<BlockIQ3XXS>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let input_buf = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_buf = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Test our kernel
    let my_kernel = my_lib
        .get_function("kernel_mul_mm_iq3_xxs_f32", None)
        .expect("Failed to get my kernel");

    let my_pipeline = device
        .new_compute_pipeline_state_with_function(&my_kernel)
        .expect("Failed to create my pipeline");

    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&my_pipeline);

    unsafe {
        let nb = 1u32;
        encoder.set_bytes(0, 4, &nb as *const u32 as *const c_void);
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&input_buf), 0);
        encoder.set_buffer(3, Some(&output_buf), 0);

        let threadgroup_size = MTLSize { width: 32, height: 1, depth: 1 };
        let threadgroup_count = MTLSize { width: 1, height: 1, depth: 1 };

        encoder.set_threadgroup_memory_length(0, 8192);
        encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
    }
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let my_result = unsafe {
        std::slice::from_raw_parts(output_buf.contents() as *const f32, m * n)
    };

    // Test llama.cpp kernel
    let llama_kernel = llama_lib
        .get_function("kernel_mul_mm_iq3_xxs_f32", None)
        .expect("Failed to get llama kernel");

    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_kernel)
        .expect("Failed to create llama pipeline");

    let llama_output_buf = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&llama_pipeline);

    unsafe {
        let nb = 1u32;
        encoder.set_bytes(0, 4, &nb as *const u32 as *const c_void);
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&input_buf), 0);
        encoder.set_buffer(3, Some(&llama_output_buf), 0);

        let threadgroup_size = MTLSize { width: 32, height: 1, depth: 1 };
        let threadgroup_count = MTLSize { width: 1, height: 1, depth: 1 };

        encoder.set_threadgroup_memory_length(0, 8192);
        encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
    }
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let llama_result = unsafe {
        std::slice::from_raw_parts(
            llama_output_buf.contents() as *const f32,
            m * n,
        )
    };

    // Compare
    println!("My result:     {:.6}", my_result[0]);
    println!("llama result:  {:.6}", llama_result[0]);
    println!("Diff:          {:.12}", (my_result[0] - llama_result[0]).abs());

    if (my_result[0] - llama_result[0]).abs() < 1e-6 {
        println!("✓ PASS");
    } else {
        println!("✗ FAIL");
    }
}
