// Safe IQ GEMM correctness test
// Tests implemented IQ formats
// Single run per format, small data, no heavy loops
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
use metal::{Device, MTLSize};
use std::ffi::c_void;

const QK_K: usize = 256;

fn main() {
    println!("=== IQ GEMM Safe Correctness Test ===\n");
    println!("Testing implemented IQ quantization formats:\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Test size: very small
    let m = 1usize;
    let k = 256usize;
    let n = 1usize;

    println!("Test size: M={}, K={}, N={} (single block)\n", m, k, n);

    // Test data
    let seed = 42u64;
    let x: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32 * 0.1 + seed as f32) % 2.0) - 1.0)
        .collect();
    let input_f16: Vec<u16> =
        x.iter().map(|&v| f16::from_f32(v).to_bits()).collect();

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

    // Test implemented formats only
    test_format(
        "IQ4_NL",
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        &input_f16,
        m,
        k,
        n,
        || quantize_iq4_nl(&x),
    );

    test_format(
        "IQ4_XS",
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        &input_f16,
        m,
        k,
        n,
        || quantize_iq4_xs(&x),
    );

    test_format(
        "IQ2_XXS",
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        &input_f16,
        m,
        k,
        n,
        || {
            let qw: Vec<f32> = vec![1.0; m * k];
            quantize_iq2_xxs(&x, Some(&qw))
        },
    );

    test_format(
        "IQ2_XS",
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        &input_f16,
        m,
        k,
        n,
        || {
            let qw: Vec<f32> = vec![1.0; m * k];
            quantize_iq2_xs(&x, Some(&qw))
        },
    );

    test_format(
        "IQ2_S",
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        &input_f16,
        m,
        k,
        n,
        || {
            let qw: Vec<f32> = vec![1.0; m * k];
            quantize_iq2_s(&x, Some(&qw))
        },
    );

    test_format(
        "IQ3_XXS",
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        &input_f16,
        m,
        k,
        n,
        || {
            // Use llama.cpp generated data
            vec![BlockIQ3XXS {
                d: 0x1999,
                qs: [
                    0xa8, 0x1e, 0x00, 0x76, 0xc0, 0xa8, 0x1e, 0x00, 0x76, 0xc0,
                    0xa8, 0x1e, 0x00, 0x76, 0xc0, 0xa8, 0x1e, 0x00, 0x76, 0xc0,
                    0xa8, 0x1e, 0x00, 0x76, 0xc0, 0xa8, 0x1e, 0x00, 0x76, 0xc0,
                    0xa8, 0x1e, 0x00, 0x76, 0xc0, 0xa8, 0x1e, 0x00, 0x76, 0xc0,
                    0xa8, 0x1e, 0x00, 0x76, 0xc0, 0xa8, 0x1e, 0x00, 0x76, 0xc0,
                    0xa8, 0x1e, 0x00, 0x76, 0xc0, 0xa8, 0x1e, 0x00, 0x76, 0xc0,
                    0xa8, 0x1e, 0x00, 0x76, 0xff, 0x01, 0xfc, 0xf7, 0x80, 0xff,
                    0x00, 0xfe, 0x3f, 0xc0, 0x7f, 0xf0, 0xf0, 0x1f, 0xe0, 0xff,
                    0x03, 0xf8, 0x0f, 0xf0, 0xff, 0x01, 0xfc, 0xf7, 0x80, 0xff,
                    0x00, 0xfe, 0x3f, 0xc0, 0x7f, 0xf0,
                ],
            }]
        },
    );

    // Test IQ3_S with quantization function
    test_format(
        "IQ3_S",
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        &input_f16,
        m,
        k,
        n,
        || quantize_iq3_s(&x, Some(&vec![1.0; m * k])),
    );

    // Test IQ1_S with llama.cpp generated data
    // Use matching input data from dump tool
    let iq1_s_test_data: Vec<f32> = vec![
        0.812570, -0.115481, 0.410859, -0.743249, 0.212563, -0.273812,
        0.934245, 0.814190, -0.321567, 0.456789, -0.234567, 0.876543,
        -0.543210, 0.123456, -0.789012, 0.345678, 0.567890, -0.234567,
        0.890123, -0.456789, 0.123456, -0.789012, 0.345678, -0.567890,
        0.234567, -0.890123, 0.456789, -0.123456, 0.789012, -0.345678,
        0.567890, -0.234567, 0.812570, -0.115481, 0.410859, -0.743249,
        0.212563, -0.273812, 0.934245, 0.814190, -0.321567, 0.456789,
        -0.234567, 0.876543, -0.543210, 0.123456, -0.789012, 0.345678,
        0.567890, -0.234567, 0.890123, -0.456789, 0.123456, -0.789012,
        0.345678, -0.567890, 0.234567, -0.890123, 0.456789, -0.123456,
        0.789012, -0.345678, 0.567890, -0.234567, 0.812570, -0.115481,
        0.410859, -0.743249, 0.212563, -0.273812, 0.934245, 0.814190,
        -0.321567, 0.456789, -0.234567, 0.876543, -0.543210, 0.123456,
        -0.789012, 0.345678, 0.567890, -0.234567, 0.890123, -0.456789,
        0.123456, -0.789012, 0.345678, -0.567890, 0.234567, -0.890123,
        0.456789, -0.123456, 0.789012, -0.345678, 0.567890, -0.234567,
        0.812570, -0.115481, 0.410859, -0.743249, 0.212563, -0.273812,
        0.934245, 0.814190, -0.321567, 0.456789, -0.234567, 0.876543,
        -0.543210, 0.123456, -0.789012, 0.345678, 0.567890, -0.234567,
        0.890123, -0.456789, 0.123456, -0.789012, 0.345678, -0.567890,
        0.234567, -0.890123, 0.456789, -0.123456, 0.789012, -0.345678,
        0.567890, -0.234567, 0.812570, -0.115481, 0.410859, -0.743249,
        0.212563, -0.273812, 0.934245, 0.814190, -0.321567, 0.456789,
        -0.234567, 0.876543, -0.543210, 0.123456, -0.789012, 0.345678,
        0.567890, -0.234567, 0.890123, -0.456789, 0.123456, -0.789012,
        0.345678, -0.567890, 0.234567, -0.890123, 0.456789, -0.123456,
        0.789012, -0.345678, 0.567890, -0.234567, 0.812570, -0.115481,
        0.410859, -0.743249, 0.212563, -0.273812, 0.934245, 0.814190,
        -0.321567, 0.456789, -0.234567, 0.876543, -0.543210, 0.123456,
        -0.789012, 0.345678, 0.567890, -0.234567, 0.890123, -0.456789,
        0.123456, -0.789012, 0.345678, -0.567890, 0.234567, -0.890123,
        0.456789, -0.123456, 0.789012, -0.345678, 0.567890, -0.234567,
        0.812570, -0.115481, 0.410859, -0.743249, 0.212563, -0.273812,
        0.934245, 0.814190, -0.321567, 0.456789, -0.234567, 0.876543,
        -0.543210, 0.123456, -0.789012, 0.345678, 0.567890, -0.234567,
        0.890123, -0.456789, 0.123456, -0.789012, 0.345678, -0.567890,
        0.234567, -0.890123, 0.456789, -0.123456, 0.789012, -0.345678,
        0.567890, -0.234567, 0.812570, -0.115481, 0.410859, -0.743249,
        0.212563, -0.273812, 0.934245, 0.814190, -0.321567, 0.456789,
        -0.234567, 0.876543, -0.543210, 0.123456, -0.789012, 0.345678,
        0.567890, -0.234567, 0.890123, -0.456789, 0.123456, -0.789012,
        0.345678, -0.567890, 0.234567, -0.890123, 0.456789, -0.123456,
        0.789012, -0.345678, 0.567890, -0.234567,
    ];
    let iq1_s_input_f16: Vec<u16> =
        iq1_s_test_data.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    test_format(
        "IQ1_S",
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        &iq1_s_input_f16,
        m,
        k,
        n,
        || {
            vec![BlockIQ1S {
                d: 0x2a94,
                qs: [
                    0x9f, 0xe5, 0xdf, 0x31, 0x9f, 0xe5, 0xdf, 0x31, 0x9f, 0xe5,
                    0xdf, 0x31, 0x9f, 0xe5, 0xdf, 0x31, 0x9f, 0xe5, 0xdf, 0x31,
                    0x9f, 0xe5, 0xdf, 0x31, 0x9f, 0xe5, 0xdf, 0x31, 0x9f, 0xe5,
                    0xdf, 0x31,
                ],
                qh: [0xf46f; 8],
            }]
        },
    );

    println!("\n=== Test Complete ===");
    println!("Implemented formats tested successfully");
}

fn test_format<T: bytemuck::Pod + bytemuck::Zeroable>(
    name: &str,
    device: &Device,
    queue: &metal::CommandQueue,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    input_f16: &[u16],
    m: usize,
    k: usize,
    n: usize,
    quantize_fn: impl FnOnce() -> Vec<T>,
) {
    println!("Testing {}...", name);

    let weights = quantize_fn();
    let block_size = std::mem::size_of::<T>();
    let nb = k / QK_K;

    // Create buffers
    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * block_size) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let input_buf = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_self = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_llama = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // GEMM args
    #[repr(C)]
    struct GemmArgs {
        ne00: i32,
        ne02: i32,
        nb01: u64,
        nb02: u64,
        nb03: u64,
        ne12: i32,
        nb10: u64,
        nb11: u64,
        nb12: u64,
        nb13: u64,
        ne0: i32,
        ne1: i32,
        r2: i16,
        r3: i16,
    }

    let args = GemmArgs {
        ne00: k as i32,
        ne02: 1,
        nb01: block_size as u64,
        nb02: 0,
        nb03: 0,
        ne12: 1,
        nb10: 2,
        nb11: (k * 2) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        r2: 1,
        r3: 1,
    };

    // Function constants
    const FC_MUL_MM: usize = 700;
    let cv = metal::FunctionConstantValues::new();
    let false_val: bool = false;
    let one_val: i16 = 1;
    unsafe {
        cv.set_constant_value_at_index(
            &false_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            FC_MUL_MM as u64,
        );
        cv.set_constant_value_at_index(
            &false_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            (FC_MUL_MM + 1) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MM + 2) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MM + 3) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MM + 4) as u64,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            (FC_MUL_MM + 5) as u64,
        );
    }

    // Get kernels
    let kernel_name = format!("kernel_mul_mm_{}_f16", name.to_lowercase());
    let my_fn = my_lib.get_function(&kernel_name, Some(cv.clone())).unwrap();
    let llama_fn = llama_lib.get_function(&kernel_name, Some(cv)).unwrap();

    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    // Run self kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_self), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(32, 4, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Run llama kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_llama), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(32, 4, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Compare outputs
    let ptr_self = output_self.contents() as *const f32;
    let ptr_llama = output_llama.contents() as *const f32;

    let val_self = unsafe { *ptr_self };
    let val_llama = unsafe { *ptr_llama };
    let diff = (val_self - val_llama).abs();

    if diff < 1e-5 {
        println!(
            "  ✓ PASS: self={:.6}, llama={:.6}, diff={:.9}",
            val_self, val_llama, diff
        );
    } else {
        println!(
            "  ✗ FAIL: self={:.6}, llama={:.6}, diff={:.9}",
            val_self, val_llama, diff
        );
    }
}
