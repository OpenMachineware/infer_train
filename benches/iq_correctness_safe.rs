// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Minimal IQ GEMM correctness test - small data to avoid GPU crash
// Tests self-written gemm.metal vs llama.cpp mul_mm_template.metal

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

fn gen_iq4_nl(m: usize, k: usize, seed: u64) -> Vec<BlockIQ4NL> {
    let nb = k / 32;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockIQ4NL { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

#[repr(C)]
struct GemmArgsMv {
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
    println!("=== Minimal IQ GEMM Correctness Test ===");

    let device = Device::system_default().expect("No Metal device");

    // Very small test: M=64, K=256, N=1
    let m = 64usize;
    let k = 256usize;
    let n = 1usize;
    let seed = 42u64;

    println!("M={}, K={}, N={} (minimal test)\n", m, k, n);

    // Load self-written gemm.metal
    let gemm_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");
    let ggml_common = include_str!("../src/ops/metal/blocks.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_DECL_METAL\n{}", ggml_common);
    let iq_grid_tables = include_str!("../src/ops/metal/iq_grid_tables.h");

    let iq_extra_tables = r#"
constexpr constant static uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
constexpr constant static int8_t ksigns_iq2xs[128] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,
     1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,
     1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,
     1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,
     1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
};
"#;

    let my_combined = format!(
        "{}\n{}\n{}\n{}",
        ggml_common_metal, iq_extra_tables, iq_grid_tables, gemm_source
    );

    let compile_opts = metal::CompileOptions::new();
    let my_lib = device
        .new_library_with_source(&my_combined, &compile_opts)
        .expect("Failed to compile self-written GEMM library");

    // Load pre-compiled llama.cpp mul_mm_template.metallib
    let llama_lib_path =
        std::path::Path::new("src/ops/metal/mul_mm_template.metallib");
    let llama_lib = device
        .new_library_with_file(llama_lib_path)
        .expect("Failed to load pre-compiled llama.cpp template library");

    // Generate input
    let input_f16 = gen_f16_input(k, n, seed);

    // Test IQ4_NL only (simplest, no grid table dependency)
    println!("--- Testing IQ4_NL ---");
    let weights = gen_iq4_nl(m, k, seed);

    // Create arguments
    let args_mv = GemmArgsMv {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 18, // block size for IQ4_NL
        nb01: (k / 32 * 18) as u64,
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
    let args_mm = GemmArgsMm {
        ne00: k as i32,
        ne02: 1,
        nb01: (k / 32 * 18) as u64,
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
    const FC_MUL_MV: usize = 600;
    let cv_my = metal::FunctionConstantValues::new();
    let nsg_val: i16 = 4; // 4 SIMD groups (tg_size.y = 4)
    let one_val: i16 = 1;
    cv_my.set_constant_value_at_index(
        &nsg_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        FC_MUL_MV as u64, // FC_nsg = 600
    );
    cv_my.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MV + 2) as u64, // FC_ne12 = 602
    );
    cv_my.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MV + 3) as u64, // FC_r2 = 603
    );
    cv_my.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_MUL_MV + 4) as u64, // FC_r3 = 604
    );

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

    let my_fn =
        my_lib.get_function("kernel_mul_mm_iq4_nl_f32", Some(cv_my)).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();

    let llama_fn = llama_lib
        .get_function("kernel_mul_mm_iq4_nl_f32", Some(cv_llama))
        .unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    // Create buffers
    let queue = device.new_command_queue();
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (m * k / 32 * 18) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (k * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_a = device.new_buffer(
        (m * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_b = device.new_buffer(
        (m * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Dispatch
    // Each threadgroup processes NSG * nr0 = 4 * 2 = 8 rows
    // Rows are indexed by tgpig.x, so we need threadgroups in X dimension
    let nsg: usize = 4;
    let nr0: usize = 2;
    let tg_size = MTLSize::new(32, nsg as u64, 1);
    let tg_count =
        MTLSize::new(((m + nsg * nr0 - 1) / (nsg * nr0)) as u64, 1, 1);

    // Run self-written kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMv>() as u64,
            &args_mv as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_a), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Run llama.cpp kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            &args_mm as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_b), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Compare results
    let ptr_a = output_a.contents() as *const f32;
    let ptr_b = output_b.contents() as *const f32;
    let slice_a = unsafe { std::slice::from_raw_parts(ptr_a, m) };
    let slice_b = unsafe { std::slice::from_raw_parts(ptr_b, m) };

    let non_zero_a: usize = slice_a.iter().filter(|&&x| x != 0.0).count();
    let non_zero_b: usize = slice_b.iter().filter(|&&x| x != 0.0).count();

    let max_diff = slice_a
        .iter()
        .zip(slice_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("  Non-zero outputs: self={}, llama={}", non_zero_a, non_zero_b);
    println!("  Max difference: {:.6}", max_diff);

    if non_zero_a == 0 || non_zero_b == 0 {
        println!("  ERROR: One kernel produced all zeros!");
    } else if max_diff < 1e-3 {
        println!("  PASS: Outputs match within tolerance");
    } else {
        println!("  FAIL: Outputs differ significantly");
        // Print first few values
        for i in 0..5.min(m) {
            println!(
                "    [{}] self={:.4}, llama={:.4}, diff={:.4}",
                i,
                slice_a[i],
                slice_b[i],
                (slice_a[i] - slice_b[i]).abs()
            );
        }
    }

    println!("\nDone.");
}
