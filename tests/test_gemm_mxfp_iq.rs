// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Quick test for MXFP4 and IQ4_NL GEMM kernels

use half::f16;
use infer_train::quant::types::*;
use metal::{Device, MTLSize};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::ffi::c_void;
use std::time::Instant;

const QK: usize = 32;

fn gen_f32_matrix(k: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect()
}

// MXFP4 block: e8m0 scale + 16 bytes of 4-bit indices
fn gen_mxfp4(m: usize, k: usize, s: u64) -> Vec<u8> {
    let nb = k / QK;
    let mut rng = StdRng::seed_from_u64(s);
    let mut result = Vec::with_capacity(m * nb * 17);
    for _ in 0..m * nb {
        // E8M0 scale byte (exponent)
        result.push(127 + (rng.r#gen::<u8>() % 16)); // exponent around 127
        // 16 bytes of 4-bit indices
        for _ in 0..16 {
            result.push(rng.r#gen::<u8>());
        }
    }
    result
}

// IQ4_NL block: half scale + 16 bytes of 4-bit indices
fn gen_iq4_nl(m: usize, k: usize, s: u64) -> Vec<u8> {
    let nb = k / QK;
    let mut rng = StdRng::seed_from_u64(s);
    let mut result = Vec::with_capacity(m * nb * 18);
    for _ in 0..m * nb {
        // Scale (2 bytes for half)
        let d = f16::from_f32(1.0).to_bits();
        result.push((d & 0xFF) as u8);
        result.push((d >> 8) as u8);
        // 16 bytes of 4-bit indices
        for _ in 0..16 {
            result.push(rng.r#gen::<u8>());
        }
    }
    result
}

#[repr(C)]
struct GemmArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
}

fn test_kernel(
    name: &str,
    device: &Device,
    weights: &[u8],
    block_size: usize,
    input: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    let lib_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");
    let compile_opts = metal::CompileOptions::new();

    let lib = match device.new_library_with_source(lib_source, &compile_opts) {
        Ok(l) => l,
        Err(e) => {
            println!("  Failed to compile: {}", e);
            return false;
        }
    };

    let cv = metal::FunctionConstantValues::new();
    let false_val: bool = false;
    let one_val: i16 = 1;
    unsafe {
        cv.set_constant_value_at_index(
            &false_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            0,
        );
        cv.set_constant_value_at_index(
            &false_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            1,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            2,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            3,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            4,
        );
        cv.set_constant_value_at_index(
            &one_val as *const _ as *const c_void,
            metal::MTLDataType::Short,
            5,
        );
    }

    let kernel_name = match name {
        "MXFP4" => "kernel_mul_mm_mxfp4_f32",
        "IQ4_NL" => "kernel_mul_mm_iq4_nl_f32",
        _ => panic!("Unknown format: {}", name),
    };

    let kernel = match lib.get_function(kernel_name, Some(cv)) {
        Ok(k) => k,
        Err(e) => {
            println!("  Failed to get kernel: {}", e);
            return false;
        }
    };

    let pipeline =
        match device.new_compute_pipeline_state_with_function(&kernel) {
            Ok(p) => p,
            Err(e) => {
                println!("  Failed to create pipeline: {}", e);
                return false;
            }
        };

    // Create buffers
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let input_f16: Vec<u16> =
        input.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let nb = k / QK;
    let args = GemmArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        ne03: 1,
        nb00: block_size as u64,
        nb01: (nb * block_size) as u64,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: n as i32,
        ne12: 1,
        ne13: 1,
        nb10: 2,
        nb11: (k * 2) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        ne2: 1,
        ne3: 1,
        nb0: 4,
        nb1: (m * 4) as u64,
        nb2: 0,
        nb3: 0,
    };

    const NR0: u64 = 64;
    const NR1: u64 = 32;
    const NSG: u64 = 4;

    let tg_size = MTLSize::new(32, NSG, 1);
    let tg_count = MTLSize::new(
        (m as u64 + NR0 * NSG - 1) / (NR0 * NSG),
        (n as u64 + NR1 - 1) / NR1,
        1,
    );

    let queue = device.new_command_queue();

    // Run once to test
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
    enc.set_buffer(3, Some(&output), 0);
    enc.set_threadgroup_memory_length(0, 8192);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    true
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let m = 4096;
    let k = 4096;
    let n = 128;
    let seed = 42;

    println!("=== MXFP4/IQ4_NL GEMM Test ===");
    println!("M={}, K={}, N={}\n", m, k, n);

    let input = gen_f32_matrix(k, n, seed);

    // Test MXFP4
    let mxfp4_weights = gen_mxfp4(m, k, seed);
    let ok = test_kernel("MXFP4", &device, &mxfp4_weights, 17, &input, m, n, k);
    println!("  MXFP4: {}", if ok { "OK" } else { "FAIL" });

    // Test IQ4_NL
    let iq4_nl_weights = gen_iq4_nl(m, k, seed);
    let ok =
        test_kernel("IQ4_NL", &device, &iq4_nl_weights, 18, &input, m, n, k);
    println!("  IQ4_NL: {}", if ok { "OK" } else { "FAIL" });

    println!("\nDone.");
}
