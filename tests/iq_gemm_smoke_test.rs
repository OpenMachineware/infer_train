// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Smoke test for IQ GEMM kernels - minimal test to verify kernels don't hang

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

fn gen_iq1_s(m: usize, k: usize, seed: u64) -> Vec<BlockIQ1S> {
    let nb = k / QK_K;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 32];
            let mut qh = [0u16; 8];
            for j in 0..32 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..8 {
                qh[j] = rng.r#gen::<u16>();
            }
            BlockIQ1S { d: f16::from_f32(1.0).to_bits(), qs, qh }
        })
        .collect()
}

// Argument structures
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GemmArgs {
    ne00: i32, // K
    ne01: i32, // M
    ne02: i32,
    nb00: u64, // block size
    nb01: u64, // row stride
    nb02: u64,
    nb03: u64,
    ne11: i32, // K
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64, // row stride for B
    nb12: u64,
    nb13: u64,
    ne0: i32, // M
    ne1: i32, // N
    r2: i32,
    r3: i32,
}

fn main() {
    println!("=== IQ GEMM Smoke Test ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Test with dimensions close to what caused the hang
    let m = 4096usize;
    let k = 4096usize;
    let n = 1usize;
    let seed = 42u64;

    println!("M={}, K={}, N={} (minimal test)", m, k, n);

    // Compile gemm.metal
    let gemm_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");
    let ggml_common = include_str!("../src/ops/metal/include/blocks.h");
    let ggml_common_metal = format!(
        "#define GGML_COMMON_DECL_METAL\n#define GGML_COMMON_IMPL_METAL\n{}",
        ggml_common
    );
    let my_combined = format!("{}\n{}", ggml_common_metal, gemm_source);

    let compile_opts = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(&my_combined, &compile_opts)
        .expect("Failed to compile GEMM library");

    // Function constants (FC_nsg = 600, etc.)
    const FC_NSG: usize = 600;
    let cv = metal::FunctionConstantValues::new();
    let nsg_val: i16 = 4;
    let one_val: i16 = 1;
    cv.set_constant_value_at_index(
        &nsg_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        FC_NSG as u64,
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_NSG + 2) as u64, // ne12
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_NSG + 3) as u64, // r2
    );
    cv.set_constant_value_at_index(
        &one_val as *const _ as *const c_void,
        metal::MTLDataType::Short,
        (FC_NSG + 4) as u64, // r3
    );

    // Get function
    let func = lib
        .get_function("kernel_mul_mm_iq1_s_f32", Some(cv))
        .expect("Failed to get function");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .expect("Failed to create pipeline");

    // Generate data
    let input_f16 = gen_f16_input(k, n, seed);
    let weights = gen_iq1_s(m, k, seed);

    // Create buffers
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * std::mem::size_of::<BlockIQ1S>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_buffer = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Arguments
    let block_bytes = std::mem::size_of::<BlockIQ1S>();
    let blocks_per_row = k / QK_K;
    let args = GemmArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: block_bytes as u64,
        nb01: (blocks_per_row * block_bytes) as u64,
        nb02: 0,
        nb03: 0,
        ne11: k as i32,
        ne12: 1,
        ne13: 1,
        nb10: 2,
        nb11: (k * 2) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        r2: 1,
        r3: 1,
    };

    // Dispatch - each threadgroup covers NR0*NSG rows
    // Kernel uses tgpig.x for row dimension (r0), tgpig.y for column dimension (r1)
    // NR0 = 4 (from kernel template), NSG = 4 (simdgroups)
    let nr0 = 4; // From kernel template parameter
    let nsg = 4; // From function constant
    let rows_per_tg = nr0 * nsg; // 16 rows per threadgroup
    let tg_size = MTLSize::new(32, nsg as u64, 1);
    // tgpig.x = row dimension, tgpig.y = column dimension
    let tg_count = MTLSize::new(
        ((m + rows_per_tg - 1) / rows_per_tg) as u64, // rows in X
        n as u64,                                     // columns in Y
        1,
    );

    println!("rows_per_tg = {}", rows_per_tg);
    println!("Threadgroups: {:?}", tg_count);
    println!("Threadgroup size: {:?}", tg_size);

    // Run once
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<GemmArgs>() as u64,
        &args as *const _ as *const c_void,
    );
    enc.set_buffer(1, Some(&weights_buffer), 0);
    enc.set_buffer(2, Some(&input_buffer), 0);
    enc.set_buffer(3, Some(&output_buffer), 0);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();

    println!("Waiting for GPU...");
    cmd.wait_until_completed();

    // Check output
    let ptr = output_buffer.contents() as *const f32;
    let slice = unsafe { std::slice::from_raw_parts(ptr, m * n) };

    println!("\nFirst 16 output values:");
    for i in 0..16.min(m * n) {
        println!("  [{}] = {}", i, slice[i]);
    }
    if m > 16 {
        println!("\nValues 16-32:");
        for i in 16..32.min(m * n) {
            println!("  [{}] = {}", i, slice[i]);
        }
    }

    let non_zero: usize = slice.iter().filter(|&&x| x != 0.0).count();
    println!("\nNon-zero values: {}/{}", non_zero, m * n);

    if non_zero > 0 {
        println!("\n✓ Kernel executed successfully!");
    } else {
        println!("\n✗ All outputs are zero - kernel may have issues");
    }
}
