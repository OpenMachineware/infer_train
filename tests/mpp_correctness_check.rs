// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Correctness check for MPP tensor API kernels
// Verifies output matches SIMD version before benchmarking

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::ffi::c_void;

const QK_K: usize = 256;

fn gen_f16_input(k: usize, n: usize, seed: u64) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect()
}

fn gen_iq3_xxs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ3XXS> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 96];
            for j in 0..96 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockIQ3XXS { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
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
    println!("=== MPP Correctness Check ===");
    println!("Testing if MPP tensor API produces same output as SIMD");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Small dimensions for quick test
    let m = 64usize;
    let k = 256usize;
    let n = 16usize;
    let seed = 42u64;

    println!("M={}, K={}, N={}", m, k, n);

    // Load libraries
    let simd_lib = device
        .new_library_with_file("src/ops/metal/mul_mm_template.metallib")
        .expect("Failed to load SIMD library");

    let mpp_lib = device
        .new_library_with_file("src/ops/metal/mul_mm_llama_tensor.metallib")
        .expect("Failed to load MPP library");

    // Function constants
    const FC_MUL_MM: usize = 700;
    let false_val: bool = false;
    let one_val: i16 = 1;

    let cv = metal::FunctionConstantValues::new();
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

    // Try to get functions
    println!("\nChecking SIMD kernel...");
    let simd_func =
        simd_lib.get_function("kernel_mul_mm_iq3_xxs_f32", Some(cv.clone()));
    match &simd_func {
        Ok(_) => println!("  SIMD kernel: OK"),
        Err(e) => println!("  SIMD kernel: FAILED - {:?}", e),
    }

    println!("\nChecking MPP kernel...");
    let mpp_func = mpp_lib.get_function("kernel_mul_mm_iq3_xxs_f32", Some(cv));
    match &mpp_func {
        Ok(_) => println!("  MPP kernel: OK"),
        Err(e) => {
            println!("  MPP kernel: FAILED - {:?}", e);
            println!("\nTrying to list available MPP kernels...");
            // List some kernels
            for name in &[
                "kernel_mul_mm_f32_f32",
                "kernel_mul_mm_f16_f32",
                "kernel_mul_mm_q4_0_f32",
            ] {
                match mpp_lib.get_function(name, None) {
                    Ok(_) => println!("    Found: {}", name),
                    Err(_) => {}
                }
            }
            return;
        }
    }

    // Create pipelines
    let simd_pipe = device
        .new_compute_pipeline_state_with_function(&simd_func.unwrap())
        .unwrap();
    let mpp_pipe = device
        .new_compute_pipeline_state_with_function(&mpp_func.unwrap())
        .unwrap();

    // Generate test data
    let weights = gen_iq3_xxs(m, k, seed);
    let input = gen_f16_input(k, n, seed);

    let args = GemmArgsMm {
        ne00: k as i32,
        ne02: 1,
        nb01: (k / 256 * 98) as u64,
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

    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * std::mem::size_of::<BlockIQ3XXS>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (input.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_simd = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_mpp = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let tg_size = MTLSize::new(128, 1, 1);
    let tg_count = MTLSize::new((n + 63) as u64 / 64, (m + 63) as u64 / 64, 1);

    // Run SIMD
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&simd_pipe);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_simd), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        println!("\nSIMD kernel executed");
    }

    // Run MPP
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&mpp_pipe);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_mpp), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        println!("MPP kernel executed");
    }

    // Compare outputs
    let ptr_simd = output_simd.contents() as *const f32;
    let ptr_mpp = output_mpp.contents() as *const f32;
    let slice_simd = unsafe { std::slice::from_raw_parts(ptr_simd, m * n) };
    let slice_mpp = unsafe { std::slice::from_raw_parts(ptr_mpp, m * n) };

    let mut max_diff = 0.0f32;
    let mut diff_count = 0;
    for (i, (a, b)) in slice_simd.iter().zip(slice_mpp.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 1e-4 {
            diff_count += 1;
            if diff_count <= 5 {
                println!(
                    "  diff at [{}]: SIMD={} MPP={} diff={}",
                    i, a, b, diff
                );
            }
        }
    }

    println!("\n=== Results ===");
    println!("max_diff: {}", max_diff);
    println!("diff_count (>1e-4): {} / {}", diff_count, m * n);

    if max_diff < 1e-4 {
        println!("PASS: MPP output matches SIMD");
    } else {
        println!("FAIL: MPP output differs from SIMD");
    }
}
