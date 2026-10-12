// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test MV correctness with larger sizes
use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use std::ffi::c_void;
use std::time::Instant;

fn main() {
    println!("=== MV Correctness Test (Large) ===");
    let device = Device::system_default().expect("No Metal device");

    // Load gemm.metal (MV kernel)
    let gemm_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");
    let ggml_common = include_str!("../src/ops/metal/blocks.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_DECL_METAL\n{}", ggml_common);
    let iq_grid_tables = include_str!("../src/ops/metal/iq_grid_tables.h");

    let combined =
        format!("{}\n{}\n{}", ggml_common_metal, iq_grid_tables, gemm_source);
    let compile_opts = metal::CompileOptions::new();
    let my_lib = device
        .new_library_with_source(&combined, &compile_opts)
        .expect("Failed to compile gemm.metal");

    // Load llama.cpp library
    let llama_lib_path =
        std::path::Path::new("src/ops/metal/mul_mm_template.metallib");
    let llama_lib = device
        .new_library_with_file(llama_lib_path)
        .expect("Failed to load llama.cpp template library");

    // Test Q4_0 MV: M=4096, K=4096, N=1
    let m = 4096usize;
    let k = 4096usize;
    let n = 1usize;

    println!("M={}, K={}, N={}", m, k, n);

    // Generate random test data
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);

    let nb = k / 32;
    let weights: Vec<BlockQ4_0> = (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockQ4_0 { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect();

    let input_f16: Vec<u16> = (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect();

    // MV args
    #[repr(C)]
    struct MvArgs {
        ne00: i32,
        ne01: i32,
        ne02: i32,
        nb00: u64,
        nb01: u64,
        nb02: u64,
        nb03: u64,
        ne10: i32,
        ne11: i32,
        ne12: i32,
        nb10: u64,
        nb11: u64,
        nb12: u64,
        nb13: u64,
        ne0: i32,
        ne1: i32,
        nr0: i32,
        r2: i16,
        r3: i16,
    }

    let args = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01: 18,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: n as i32,
        ne12: 1,
        nb10: 2,
        nb11: (k * 2) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        nr0: 1,
        r2: 1,
        r3: 1,
    };

    let queue = device.new_command_queue();

    // Function constants for MV kernel
    const FC_MUL_MV: usize = 600;
    let cv = metal::FunctionConstantValues::new();
    let nsg_val: i16 = 4;
    let one_val: i16 = 1;
    cv.set_constant_value_at_index(
        &nsg_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        FC_MUL_MV as u64,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MV + 2) as u64,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MV + 3) as u64,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MV + 4) as u64,
    );

    let my_pipeline = device
        .new_compute_pipeline_state_with_function(
            &my_lib
                .get_function("kernel_mul_mm_q4_0_f32", Some(cv.clone()))
                .unwrap(),
        )
        .unwrap();

    // Function constants for llama.cpp (FC=700)
    const FC_MUL_MM: usize = 700;
    let cv_llama = metal::FunctionConstantValues::new();
    let false_val: bool = false;
    cv_llama.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        FC_MUL_MM as u64,
    );
    cv_llama.set_constant_value_at_index(
        &false_val as *const _ as *const c_void,
        metal::MTLDataType::Bool,
        (FC_MUL_MM + 1) as u64,
    );
    cv_llama.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 2) as u64,
    );
    cv_llama.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 3) as u64,
    );
    cv_llama.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 4) as u64,
    );
    cv_llama.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MM + 5) as u64,
    );

    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(
            &llama_lib
                .get_function("kernel_mul_mm_q4_0_f32", Some(cv_llama))
                .unwrap(),
        )
        .unwrap();

    // Create buffers
    let weights_bytes: &[u8] = bytemuck::cast_slice(&weights);
    let weights_buffer = device.new_buffer_with_data(
        weights_bytes.as_ptr() as *const c_void,
        weights_bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_self = device.new_buffer(
        (m * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_llama = device.new_buffer(
        (m * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Dispatch
    let nsg = 4;
    let nr0 = 1;
    let rows_per_tg = nsg * nr0;
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count =
        MTLSize::new(1, ((m + rows_per_tg - 1) / rows_per_tg) as u64, 1);

    // Warmup
    for _ in 0..5 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
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

    // Run self-written MV
    let start = Instant::now();
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
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
    let self_us = start.elapsed().as_micros();

    // Run llama.cpp
    let start = Instant::now();
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        // Note: llama.cpp uses different args structure for mul_mm
        // For MV, we can reuse the MvArgs but need to adjust
        #[repr(C)]
        struct MmArgs {
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
        let mm_args = MmArgs {
            ne00: k as i32,
            ne02: 1,
            nb01: 18,
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
        enc.set_bytes(
            0,
            std::mem::size_of::<MmArgs>() as u64,
            &mm_args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_llama), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        // llama.cpp GEMM dispatch
        let llama_tg_size = MTLSize::new(128, 4, 1);
        let llama_tg_count =
            MTLSize::new(((n + 31) / 32) as u64, ((m + 63) / 64) as u64, 1);
        enc.dispatch_thread_groups(llama_tg_count, llama_tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let llama_us = start.elapsed().as_micros();

    // Compare results
    let ptr_self = output_self.contents() as *const f32;
    let ptr_llama = output_llama.contents() as *const f32;
    let slice_self = unsafe { std::slice::from_raw_parts(ptr_self, m) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m) };

    let non_zero_self: usize = slice_self.iter().filter(|&&x| x != 0.0).count();
    let non_zero_llama: usize =
        slice_llama.iter().filter(|&&x| x != 0.0).count();

    let max_diff = slice_self
        .iter()
        .zip(slice_llama.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let sum_self: f64 = slice_self.iter().map(|&x| x as f64).sum();
    let sum_llama: f64 = slice_llama.iter().map(|&x| x as f64).sum();

    println!("\nResults:");
    println!("  Self-written MV: {} µs", self_us);
    println!("  llama.cpp GEMM: {} µs", llama_us);
    println!("  Ratio: {:.1}%", self_us as f64 / llama_us as f64 * 100.0);
    println!("\nCorrectness:");
    println!("  Non-zero (self): {}", non_zero_self);
    println!("  Non-zero (llama): {}", non_zero_llama);
    println!("  Max diff: {:.2}", max_diff);
    println!("  Sum (self): {:.2}", sum_self);
    println!("  Sum (llama): {:.2}", sum_llama);
    println!(
        "  Status: {}",
        if max_diff < 1.0 { "✅ PASS" } else { "❌ FAIL" }
    );
}
