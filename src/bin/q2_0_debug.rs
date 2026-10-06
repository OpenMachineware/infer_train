// Debug Q2_0 GEMM with known values
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use half::f16;
use metal::{Device, MTLSize};
use std::ffi::c_void;

#[repr(C, packed)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlockQ20 {
    d: u16,
    qs: [u8; 16],
}

#[repr(C)]
struct GemmArgsMm {
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

fn main() {
    println!("=== Q2_0 Debug Test ===\n");

    let device = Device::system_default().expect("No Metal device");
    let my_lib = device
        .new_library_with_file(std::path::Path::new(
            "shaders/mul_mm_standalone.metallib",
        ))
        .unwrap();
    let llama_lib = device
        .new_library_with_file(std::path::Path::new(
            "shaders/mul_mm_template.metallib",
        ))
        .unwrap();

    // Test with M=64, K=64, N=1 (1 block per row)
    // Simpler test to verify dequantize is correct
    let m: usize = 64;
    let k: usize = 64; // 1 block per row
    let n: usize = 1;

    // Q2_0 block with known values
    // qs bytes: each byte has 4 elements
    // Let's use byte values where we can predict the output:
    // Byte 0x00: elements are 0,0,0,0 → values are -d, -d, -d, -d
    // Byte 0x55: elements are 1,1,1,1 → values are 0, 0, 0, 0
    // Byte 0xAA: elements are 2,2,2,2 → values are d, d, d, d
    // Byte 0xFF: elements are 3,3,3,3 → values are 2d, 2d, 2d, 2d

    let d = 0.1f32;
    let d_f16 = f16::from_f32(d).to_bits();

    // Q2_0: 64 elements per block, so K=256 needs 4 blocks per row
    let blocks_per_row = k / 64;
    let total_blocks = m * blocks_per_row;

    // Create blocks where all 64 elements have the same quantized value
    // Let's use all zeros (quantized value 0 → -d = -0.1)
    let weights: Vec<BlockQ20> = (0..total_blocks)
        .map(|_| BlockQ20 {
            d: d_f16,
            qs: [0x00; 16], // All elements are 0 → all output values should be -0.1
        })
        .collect();

    // Input is all 1.0 (so output should be sum of weights)
    let input_f16: Vec<u16> = vec![f16::from_f32(1.0).to_bits(); k];

    let queue = device.new_command_queue();

    // Setup function constants
    const FC_MUL_MM: usize = 700;
    let true_val: bool = true;
    let one_val: i16 = 1;

    let cv = metal::FunctionConstantValues::new();
    unsafe {
        cv.set_constant_value_at_index(
            &true_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            FC_MUL_MM as u64,
        );
        cv.set_constant_value_at_index(
            &true_val as *const _ as *const c_void,
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

    let my_fn = my_lib
        .get_function("kernel_mul_mm_q2_0_f32", Some(cv.clone()))
        .unwrap();
    let llama_fn =
        llama_lib.get_function("kernel_mul_mm_q2_0_f32", Some(cv)).unwrap();

    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * 18) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (k * 2) as u64,
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

    // For Q2_0: nl=4, block_bytes=18, nb=blocks_per_row
    // nb01 is byte stride between ROWS (not blocks)
    let args = GemmArgsMm {
        ne00: k as i32,
        ne02: 1,
        nb01: (blocks_per_row * 18) as u64,
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

    // The kernel uses:
    // const int r0 = tgpig.y*NR0;  // NR0=64 rows per threadgroup
    // const int r1 = tgpig.x*NR1;  // NR1=32 columns per threadgroup
    // MTLSize::new(width, height, depth) = MTLSize::new(tg_count.x, tg_count.y, tg_count.z)
    // For GEMM: NR0=64, NR1=32 (fixed in kernel template)
    //   tg_count.y = ceil(M / NR0)
    //   tg_count.x = ceil(N / NR1)
    let nr0: usize = 64; // Fixed in kernel
    let nr1: usize = 32; // Fixed in kernel
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count = MTLSize::new(
        ((n + nr1 - 1) / nr1) as u64,
        ((m + nr0 - 1) / nr0) as u64,
        1,
    );

    // Run self kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_self), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
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
            std::mem::size_of::<GemmArgsMm>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_llama), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let ptr_self = output_self.contents() as *const f32;
    let ptr_llama = output_llama.contents() as *const f32;

    // Expected output: K elements × (-0.1) × 1.0 = K * (-0.1)
    let expected = (k as f32) * (-0.1f32);

    println!("Test: All {} elements have quantized value 0 (→ -0.1)", k);
    println!("Input: all 1.0");
    println!(
        "Expected output per row: {} × (-0.1) × 1.0 = {:.1}\n",
        k, expected
    );

    let slice_self = unsafe { std::slice::from_raw_parts(ptr_self, m) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m) };

    println!("First 5 outputs:");
    for i in 0..5.min(m) {
        println!(
            "  [{}] self={:.6}, llama={:.6}, expected={:.6}, diff={:.6}",
            i,
            slice_self[i],
            slice_llama[i],
            expected,
            (slice_self[i] - expected).abs()
        );
    }

    let max_diff_self =
        slice_self.iter().map(|&x| (x - expected).abs()).fold(0.0f32, f32::max);
    let max_diff_llama = slice_llama
        .iter()
        .map(|&x| (x - expected).abs())
        .fold(0.0f32, f32::max);

    println!("Max diff (self):  {:.6}", max_diff_self);
    println!("Max diff (llama): {:.6}", max_diff_llama);

    // Now test with quantized value 2 (→ 0.1)
    let weights2: Vec<BlockQ20> = (0..total_blocks)
        .map(|_| BlockQ20 {
            d: d_f16,
            qs: [0xAA; 16], // All elements are 2 → all output values should be 0.1
        })
        .collect();

    let weights2_buffer = device.new_buffer_with_data(
        weights2.as_ptr() as *const c_void,
        (weights2.len() * 18) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Expected output: K × 0.1 × 1.0 = K * 0.1
    let expected2 = (k as f32) * 0.1f32;

    // Run self kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights2_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_self), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
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
            std::mem::size_of::<GemmArgsMm>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights2_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_llama), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    println!("\nTest: All {} elements have quantized value 2 (→ 0.1)", k);
    println!("Expected output per row: {} × 0.1 × 1.0 = {:.1}\n", k, expected2);

    println!("First 5 outputs:");
    for i in 0..5.min(m) {
        println!(
            "  [{}] self={:.6}, llama={:.6}, expected={:.6}, diff={:.6}",
            i,
            slice_self[i],
            slice_llama[i],
            expected2,
            (slice_self[i] - expected2).abs()
        );
    }

    let max_diff_self2 = slice_self
        .iter()
        .map(|&x| (x - expected2).abs())
        .fold(0.0f32, f32::max);
    let max_diff_llama2 = slice_llama
        .iter()
        .map(|&x| (x - expected2).abs())
        .fold(0.0f32, f32::max);

    println!("Max diff (self):  {:.6}", max_diff_self2);
    println!("Max diff (llama): {:.6}", max_diff_llama2);
}
