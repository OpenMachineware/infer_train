// Verify all GEMM kernels exist and work
// Q1_0, Q2_0, F32, F16
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use half::f16;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;

// Block sizes from ggml-common.h
const QK1_0: usize = 128; // 128 values per block
const QK2_0: usize = 64; // 64 values per block

// Rust block types matching Metal layout
#[repr(C, packed)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlockQ10 {
    d: u16,
    qs: [u8; QK1_0 / 8], // 16 bytes
}

#[repr(C, packed)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlockQ20 {
    d: u16,
    qs: [u8; QK2_0 / 4], // 16 bytes
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

fn test_q1_0(
    device: &Device,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    m: usize,
    k: usize,
    n: usize,
) {
    println!("\n=== Q1_0 (M={}, K={}) ===", m, k);

    let queue = device.new_command_queue();
    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);

    // Q1_0: QK1_0=128 values per block, 18 bytes (d=2 + qs=16)
    let nb = k / QK1_0;
    let weights: Vec<BlockQ10> = (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; QK1_0 / 8];
            for j in 0..QK1_0 / 8 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockQ10 { d: f16::from_f32(0.1).to_bits(), qs }
        })
        .collect();

    // Generate input
    let input_f16: Vec<u16> = (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect();

    const FC_MUL_MM: usize = 700;
    let false_val: bool = false;
    let one_val: i16 = 1;

    let cv = metal::FunctionConstantValues::new();
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

    let my_fn = my_lib
        .get_function("kernel_mul_mm_q1_0_f32", Some(cv.clone()))
        .unwrap();
    let llama_fn =
        llama_lib.get_function("kernel_mul_mm_q1_0_f32", Some(cv)).unwrap();

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

    let args = GemmArgsMm {
        ne00: k as i32,
        ne02: 1,
        nb01: (nb * 18) as u64,
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

    let nr0: usize = 64; // Fixed in kernel template
    let nr1: usize = 32; // Fixed in kernel template
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count = MTLSize::new(
        ((n + nr1 - 1) / nr1) as u64,
        ((m + nr0 - 1) / nr0) as u64,
        1,
    );

    // Run self
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

    // Run llama
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
    let slice_self = unsafe { std::slice::from_raw_parts(ptr_self, m * n) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m * n) };

    let max_diff = slice_self
        .iter()
        .zip(slice_llama.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Show first few outputs
    println!("  First 5 outputs:");
    for i in 0..5.min(m * n) {
        println!(
            "    [{}] self={:.6}, llama={:.6}, diff={:.6}",
            i,
            slice_self[i],
            slice_llama[i],
            (slice_self[i] - slice_llama[i]).abs()
        );
    }

    println!("  Max diff: {:.6}", max_diff);
    // F16 accumulation has ~0.001-0.01 precision loss
    if max_diff < 0.01 {
        println!("  ✓ PASS (within F16 precision tolerance)");
    } else {
        println!("  ✗ FAIL");
    }
}

fn test_q2_0(
    device: &Device,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    m: usize,
    k: usize,
    n: usize,
) {
    println!("\n=== Q2_0 (M={}, K={}) ===", m, k);

    let queue = device.new_command_queue();
    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);

    // Q2_0: QK2_0=64 values per block, 18 bytes (d=2 + qs=16)
    let nb = k / QK2_0;
    let weights: Vec<BlockQ20> = (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; QK2_0 / 4];
            for j in 0..QK2_0 / 4 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockQ20 { d: f16::from_f32(0.1).to_bits(), qs }
        })
        .collect();

    let input_f16: Vec<u16> = (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect();

    const FC_MUL_MM: usize = 700;
    let false_val: bool = false;
    let one_val: i16 = 1;

    let cv = metal::FunctionConstantValues::new();
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

    let args = GemmArgsMm {
        ne00: k as i32,
        ne02: 1,
        nb01: (nb * 18) as u64,
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

    let nr0: usize = 64; // Fixed in kernel template
    let nr1: usize = 32; // Fixed in kernel template
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count = MTLSize::new(
        ((n + nr1 - 1) / nr1) as u64,
        ((m + nr0 - 1) / nr0) as u64,
        1,
    );

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
    let slice_self = unsafe { std::slice::from_raw_parts(ptr_self, m * n) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m * n) };

    let max_diff = slice_self
        .iter()
        .zip(slice_llama.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Show first few outputs
    println!("  First 5 outputs:");
    for i in 0..5.min(m * n) {
        println!(
            "    [{}] self={:.6}, llama={:.6}, diff={:.6}",
            i,
            slice_self[i],
            slice_llama[i],
            (slice_self[i] - slice_llama[i]).abs()
        );
    }

    println!("  Max diff: {:.6}", max_diff);
    // F16 accumulation has ~0.001-0.01 precision loss
    if max_diff < 0.01 {
        println!("  ✓ PASS (within F16 precision tolerance)");
    } else {
        println!("  ✗ FAIL");
    }
}

fn test_f32(
    device: &Device,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    m: usize,
    k: usize,
    n: usize,
) {
    println!("\n=== F32 (M={}, K={}) ===", m, k);

    let queue = device.new_command_queue();
    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);

    // F32 weights (no quantization)
    let weights_f32: Vec<f32> =
        (0..m * k).map(|_| rng.r#gen::<f32>() * 0.1).collect();

    let input_f16: Vec<u16> = (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect();

    const FC_MUL_MM: usize = 700;
    let false_val: bool = false;
    let one_val: i16 = 1;

    let cv = metal::FunctionConstantValues::new();
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

    let my_fn =
        my_lib.get_function("kernel_mul_mm_f32_f32", Some(cv.clone())).unwrap();
    let llama_fn =
        llama_lib.get_function("kernel_mul_mm_f32_f32", Some(cv)).unwrap();

    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    let weights_buffer = device.new_buffer_with_data(
        weights_f32.as_ptr() as *const c_void,
        (weights_f32.len() * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
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

    // F32: stride is k * 4 bytes (one row)
    let args = GemmArgsMm {
        ne00: k as i32,
        ne02: 1,
        nb01: (k * 4) as u64,
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

    let nr0: usize = 64; // Fixed in kernel template
    let nr1: usize = 32; // Fixed in kernel template
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count = MTLSize::new(
        ((n + nr1 - 1) / nr1) as u64,
        ((m + nr0 - 1) / nr0) as u64,
        1,
    );

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
    let slice_self = unsafe { std::slice::from_raw_parts(ptr_self, m * n) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m * n) };

    let max_diff = slice_self
        .iter()
        .zip(slice_llama.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Show first few outputs
    println!("  First 5 outputs:");
    for i in 0..5.min(m * n) {
        println!(
            "    [{}] self={:.6}, llama={:.6}, diff={:.6}",
            i,
            slice_self[i],
            slice_llama[i],
            (slice_self[i] - slice_llama[i]).abs()
        );
    }

    println!("  Max diff: {:.6}", max_diff);
    // F16 accumulation has ~0.001-0.01 precision loss
    if max_diff < 0.01 {
        println!("  ✓ PASS (within F16 precision tolerance)");
    } else {
        println!("  ✗ FAIL");
    }
}

fn test_f16(
    device: &Device,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    m: usize,
    k: usize,
    n: usize,
) {
    println!("\n=== F16 (M={}, K={}) ===", m, k);

    let queue = device.new_command_queue();
    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);

    // F16 weights
    let weights_f16: Vec<u16> = (0..m * k)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 0.1).to_bits())
        .collect();

    let input_f16: Vec<u16> = (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect();

    const FC_MUL_MM: usize = 700;
    let false_val: bool = false;
    let one_val: i16 = 1;

    let cv = metal::FunctionConstantValues::new();
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

    let my_fn =
        my_lib.get_function("kernel_mul_mm_f16_f32", Some(cv.clone())).unwrap();
    let llama_fn =
        llama_lib.get_function("kernel_mul_mm_f16_f32", Some(cv)).unwrap();

    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    let weights_buffer = device.new_buffer_with_data(
        weights_f16.as_ptr() as *const c_void,
        (weights_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
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

    // F16: stride is k * 2 bytes (one row)
    let args = GemmArgsMm {
        ne00: k as i32,
        ne02: 1,
        nb01: (k * 2) as u64,
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

    let nr0: usize = 64; // Fixed in kernel template
    let nr1: usize = 32; // Fixed in kernel template
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count = MTLSize::new(
        ((n + nr1 - 1) / nr1) as u64,
        ((m + nr0 - 1) / nr0) as u64,
        1,
    );

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
    let slice_self = unsafe { std::slice::from_raw_parts(ptr_self, m * n) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m * n) };

    let max_diff = slice_self
        .iter()
        .zip(slice_llama.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Show first few outputs
    println!("  First 5 outputs:");
    for i in 0..5.min(m * n) {
        println!(
            "    [{}] self={:.6}, llama={:.6}, diff={:.6}",
            i,
            slice_self[i],
            slice_llama[i],
            (slice_self[i] - slice_llama[i]).abs()
        );
    }

    println!("  Max diff: {:.6}", max_diff);
    // F16 accumulation has ~0.001-0.01 precision loss
    if max_diff < 0.01 {
        println!("  ✓ PASS (within F16 precision tolerance)");
    } else {
        println!("  ✗ FAIL");
    }
}

fn main() {
    println!("=== Missing GEMM Kernels Verification ===\n");

    let device = Device::system_default().expect("No Metal device");

    let my_lib = device
        .new_library_with_file(std::path::Path::new(
            "src/ops/metal/mul_mm_standalone.metallib",
        ))
        .unwrap();
    let llama_lib = device
        .new_library_with_file(std::path::Path::new(
            "src/ops/metal/mul_mm_template.metallib",
        ))
        .unwrap();

    let m = 64usize;
    let k = 256usize;
    let n = 32usize; // Use N=32 to match NR1

    test_q1_0(&device, &my_lib, &llama_lib, m, k, n);
    test_q2_0(&device, &my_lib, &llama_lib, m, k, n);
    test_f32(&device, &my_lib, &llama_lib, m, k, n);
    test_f16(&device, &my_lib, &llama_lib, m, k, n);

    println!("\n=== Summary ===");
    println!("All 4 missing kernels tested");
}
