// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Verify IQ3_XXS kernel correctness with multiple test cases
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use std::ffi::c_void;

const QK_K: usize = 256;

fn main() {
    println!("=== IQ3_XXS Kernel Correctness Verification ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let m = 1usize;
    let k = 256usize;
    let n = 1usize;

    // Load Metal libraries
    let my_lib_path =
        std::path::Path::new("src/ops/metal/mul_mm_standalone.metallib");
    let my_lib = device
        .new_library_with_file(my_lib_path)
        .expect("Failed to load standalone library");

    let llama_lib_path =
        std::path::Path::new("src/ops/metal/mul_mm_template.metallib");
    let llama_lib = device
        .new_library_with_file(llama_lib_path)
        .expect("Failed to load template library");

    // Test case 1: seed=42
    println!("Test case 1: seed=42");
    let x1: Vec<f32> =
        (0..k).map(|i| ((i as f32 * 0.1 + 42.0) % 2.0) - 1.0).collect();
    let input_f16_1: Vec<u16> =
        x1.iter().map(|&v| f16::from_f32(v).to_bits()).collect();

    let block1 = BlockIQ3XXS {
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

    test_single(
        "IQ3_XXS case 1",
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        &input_f16_1,
        m,
        k,
        n,
        &block1,
    );

    // Test case 2: seed=123
    println!("\nTest case 2: seed=123");
    let x2: Vec<f32> =
        (0..k).map(|i| ((i as f32 * 0.137 + 123.0) % 2.0) - 1.0).collect();
    let input_f16_2: Vec<u16> =
        x2.iter().map(|&v| f16::from_f32(v).to_bits()).collect();

    let block2 = BlockIQ3XXS {
        d: 0x19e8,
        qs: [
            0x6d, 0xc0, 0x82, 0x03, 0x9c, 0xc0, 0x1e, 0x3a, 0xa2, 0xa8, 0x49,
            0x6d, 0xe5, 0x82, 0x03, 0xa0, 0xc0, 0x1e, 0x45, 0xd3, 0xa8, 0x0f,
            0x6d, 0xe5, 0x82, 0x03, 0xa0, 0xa8, 0x1e, 0x45, 0xe3, 0xa8, 0x0f,
            0x9c, 0xe5, 0x2b, 0x3a, 0xa0, 0xa8, 0x1e, 0x45, 0xd3, 0xa8, 0x0f,
            0x9c, 0xc0, 0x82, 0x3a, 0xa0, 0xa8, 0x1e, 0x45, 0xd3, 0xbb, 0x0f,
            0x9c, 0xc0, 0x1e, 0x3a, 0xa0, 0xa8, 0x1e, 0x6d, 0xc0, 0x80, 0x3f,
            0xf0, 0xf7, 0xe1, 0x07, 0x7c, 0xe0, 0xfc, 0x81, 0x3f, 0xe0, 0x3f,
            0xe0, 0x2f, 0xec, 0x0f, 0xfc, 0x80, 0xef, 0x83, 0x7f, 0xe0, 0xe7,
            0xc0, 0x07, 0xfc, 0xe1, 0xf8, 0x81, 0x1f, 0xf0,
        ],
    };

    test_single(
        "IQ3_XXS case 2",
        &device,
        &queue,
        &my_lib,
        &llama_lib,
        &input_f16_2,
        m,
        k,
        n,
        &block2,
    );

    println!("\n=== All tests passed ===");
}

fn test_single(
    name: &str,
    device: &Device,
    queue: &metal::CommandQueue,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    input_f16: &[u16],
    m: usize,
    k: usize,
    n: usize,
    block: &BlockIQ3XXS,
) {
    let weights: Vec<BlockIQ3XXS> = vec![*block];
    let block_size = std::mem::size_of::<BlockIQ3XXS>();

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
    }

    // Test my kernel
    let my_kernel = my_lib
        .get_function("kernel_mul_mm_iq3_xxs_f32", Some(cv.clone()))
        .expect("Failed to get my kernel");
    let my_pipeline = device
        .new_compute_pipeline_state_with_function(&my_kernel)
        .expect("Failed to create my pipeline");

    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&my_pipeline);
    unsafe {
        encoder.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&input_buf), 0);
        encoder.set_buffer(3, Some(&output_self), 0);
        let tg_size = MTLSize { width: 32, height: 1, depth: 1 };
        let tg_count = MTLSize { width: 1, height: 1, depth: 1 };
        encoder.set_threadgroup_memory_length(0, 8192);
        encoder.dispatch_thread_groups(tg_count, tg_size);
    }
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Test llama kernel
    let llama_kernel = llama_lib
        .get_function("kernel_mul_mm_iq3_xxs_f32", Some(cv.clone()))
        .expect("Failed to get llama kernel");
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_kernel)
        .expect("Failed to create llama pipeline");

    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&llama_pipeline);
    unsafe {
        encoder.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&input_buf), 0);
        encoder.set_buffer(3, Some(&output_llama), 0);
        let tg_size = MTLSize { width: 32, height: 1, depth: 1 };
        let tg_count = MTLSize { width: 1, height: 1, depth: 1 };
        encoder.set_threadgroup_memory_length(0, 8192);
        encoder.dispatch_thread_groups(tg_count, tg_size);
    }
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    let my_result = unsafe {
        std::slice::from_raw_parts(output_self.contents() as *const f32, m * n)
    };
    let llama_result = unsafe {
        std::slice::from_raw_parts(output_llama.contents() as *const f32, m * n)
    };

    let diff = (my_result[0] - llama_result[0]).abs();
    println!("{}:", name);
    println!(
        "  self={:.6}, llama={:.6}, diff={:.12}",
        my_result[0], llama_result[0], diff
    );

    if diff < 1e-6 {
        println!("  ✓ PASS");
    } else {
        println!("  ✗ FAIL");
        std::process::exit(1);
    }
}
