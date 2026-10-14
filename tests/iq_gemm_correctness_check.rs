// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// IQ GEMM correctness check: compare self-written vs llama.cpp
// Single run per format, compare outputs - no heavy loops
// Uses REAL quantized data (not random bytes)

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::weights::iq::*;
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

fn gen_f32_weights(k: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k).map(|_| (rng.r#gen::<f32>() * 2.0 - 1.0)).collect()
}

fn gen_uniform_qweights(k: usize) -> Vec<f32> {
    vec![1.0f32; k]
}

fn gen_iq1_s(m: usize, k: usize, seed: u64) -> Vec<BlockIQ1S> {
    let x = gen_f32_weights(m * k, seed);
    let qw = gen_uniform_qweights(m * k);
    quantize_iq1_s(&x, Some(&qw))
}

fn gen_iq1_m(m: usize, k: usize, seed: u64) -> Vec<BlockIQ1M> {
    let x = gen_f32_weights(m * k, seed);
    let qw = gen_uniform_qweights(m * k);
    quantize_iq1_m(&x, Some(&qw))
}

fn gen_iq2_xxs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ2XXS> {
    let x = gen_f32_weights(m * k, seed);
    let qw = gen_uniform_qweights(m * k);
    quantize_iq2_xxs(&x, Some(&qw))
}

fn gen_iq2_xs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ2XS> {
    let x = gen_f32_weights(m * k, seed);
    let qw = gen_uniform_qweights(m * k);
    quantize_iq2_xs(&x, Some(&qw))
}

fn gen_iq2_s(m: usize, k: usize, seed: u64) -> Vec<BlockIQ2S> {
    let x = gen_f32_weights(m * k, seed);
    let qw = gen_uniform_qweights(m * k);
    quantize_iq2_s(&x, Some(&qw))
}

fn gen_iq3_xxs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ3XXS> {
    let x = gen_f32_weights(m * k, seed);
    let qw = gen_uniform_qweights(m * k);
    quantize_iq3_xxs(&x, Some(&qw))
}

fn gen_iq3_s(m: usize, k: usize, seed: u64) -> Vec<BlockIQ3S> {
    let x = gen_f32_weights(m * k, seed);
    let qw = gen_uniform_qweights(m * k);
    quantize_iq3_s(&x, Some(&qw))
}

fn gen_iq4_nl(m: usize, k: usize, seed: u64) -> Vec<BlockIQ4NL> {
    let x = gen_f32_weights(m * k, seed);
    quantize_iq4_nl(&x)
}

fn gen_iq4_xs(m: usize, k: usize, seed: u64) -> Vec<BlockIQ4XS> {
    let x = gen_f32_weights(m * k, seed);
    quantize_iq4_xs(&x)
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

fn check_correctness(
    name: &str,
    my_kernel: &str,
    llama_kernel: &str,
    device: &Device,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    weights: &[u8],
    input_f16: &[u16],
    args_mm: &GemmArgsMm,
    m: usize,
    n: usize,
    nr0: usize,
) {
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

    let my_fn = my_lib.get_function(my_kernel, Some(cv.clone())).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();

    let llama_fn =
        llama_lib.get_function(llama_kernel, Some(cv.clone())).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_fn).unwrap();

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
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

    let queue = device.new_command_queue();
    let nsg = 4;
    let rows_per_tg = nsg * nr0;
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count =
        MTLSize::new(((m + rows_per_tg - 1) / rows_per_tg) as u64, n as u64, 1);

    // Run self-written kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            args_mm as *const _ as *const c_void,
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

    // Run llama.cpp kernel
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgsMm>() as u64,
            args_mm as *const _ as *const c_void,
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

    // Compare outputs
    let ptr_self = output_self.contents() as *const f32;
    let ptr_llama = output_llama.contents() as *const f32;
    let slice_self = unsafe { std::slice::from_raw_parts(ptr_self, m * n) };
    let slice_llama = unsafe { std::slice::from_raw_parts(ptr_llama, m * n) };

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    let mut diff_count = 0;
    let threshold = 1e-4f32;

    for (i, (&a, &b)) in slice_self.iter().zip(slice_llama.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > threshold {
            diff_count += 1;
        }
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    if diff_count == 0 {
        println!("  {}: ✓ PASS (max_diff={:.6})", name, max_diff);
    } else {
        println!(
            "  {}: ❌ FAIL (diff_count={}, max_diff={:.6} at idx {})",
            name, diff_count, max_diff, max_diff_idx
        );
        // Show first few differences
        let shown = 5;
        let mut shown_count = 0;
        for (i, (&a, &b)) in
            slice_self.iter().zip(slice_llama.iter()).enumerate()
        {
            let diff = (a - b).abs();
            if diff > threshold && shown_count < shown {
                println!(
                    "    idx {}: self={:.6}, llama={:.6}, diff={:.6}",
                    i, a, b, diff
                );
                shown_count += 1;
            }
        }
    }
}

fn main() {
    println!("=== IQ GEMM Correctness Check ===");
    println!("Single run per format, compare outputs\n");

    let device = Device::system_default().expect("No Metal device");

    // Small test size to minimize risk
    let m = 64usize;
    let k = 256usize;
    let n = 1usize;
    let seed = 42u64;

    println!("M={}, K={}, N={}\n", m, k, n);

    let my_lib_path =
        std::path::Path::new("src/ops/metal/mul_mm_standalone.metallib");
    let my_lib = device
        .new_library_with_file(my_lib_path)
        .expect("Failed to load self-written library");

    let llama_lib_path =
        std::path::Path::new("src/ops/metal/mul_mm_template.metallib");
    let llama_lib = device
        .new_library_with_file(llama_lib_path)
        .expect("Failed to load llama.cpp library");

    let input_f16 = gen_f16_input(k, n, seed);

    let make_args = |k: usize,
                     m: usize,
                     n: usize,
                     block_size: usize,
                     block_bytes: usize|
     -> GemmArgsMm {
        GemmArgsMm {
            ne00: k as i32,
            ne02: 1,
            nb01: (k / block_size * block_bytes) as u64,
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
        }
    };

    println!("=== Testing 9 IQ formats ===\n");

    // IQ1_S (50 bytes)
    let weights = gen_iq1_s(m, k, seed);
    let args = make_args(k, m, n, 256, 50);
    check_correctness(
        "IQ1_S",
        "kernel_mul_mm_iq1_s_f32",
        "kernel_mul_mm_iq1_s_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        4,
    );

    // IQ1_M (56 bytes)
    let weights = gen_iq1_m(m, k, seed);
    let args = make_args(k, m, n, 256, 56);
    check_correctness(
        "IQ1_M",
        "kernel_mul_mm_iq1_m_f32",
        "kernel_mul_mm_iq1_m_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        4,
    );

    // IQ2_XXS (66 bytes)
    let weights = gen_iq2_xxs(m, k, seed);
    let args = make_args(k, m, n, 256, 66);
    check_correctness(
        "IQ2_XXS",
        "kernel_mul_mm_iq2_xxs_f32",
        "kernel_mul_mm_iq2_xxs_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        4,
    );

    // IQ2_XS (74 bytes)
    let weights = gen_iq2_xs(m, k, seed);
    let args = make_args(k, m, n, 256, 74);
    check_correctness(
        "IQ2_XS",
        "kernel_mul_mm_iq2_xs_f32",
        "kernel_mul_mm_iq2_xs_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        4,
    );

    // IQ2_S (82 bytes)
    let weights = gen_iq2_s(m, k, seed);
    let args = make_args(k, m, n, 256, 82);
    check_correctness(
        "IQ2_S",
        "kernel_mul_mm_iq2_s_f32",
        "kernel_mul_mm_iq2_s_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        4,
    );

    // IQ3_XXS (98 bytes)
    let weights = gen_iq3_xxs(m, k, seed);
    let args = make_args(k, m, n, 256, 98);
    check_correctness(
        "IQ3_XXS",
        "kernel_mul_mm_iq3_xxs_f32",
        "kernel_mul_mm_iq3_xxs_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        4,
    );

    // IQ3_S (110 bytes)
    let weights = gen_iq3_s(m, k, seed);
    let args = make_args(k, m, n, 256, 110);
    check_correctness(
        "IQ3_S",
        "kernel_mul_mm_iq3_s_f32",
        "kernel_mul_mm_iq3_s_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        4,
    );

    // IQ4_NL (34 bytes per 32)
    let weights = gen_iq4_nl(m, k, seed);
    let args = make_args(k, m, n, 32, 34);
    check_correctness(
        "IQ4_NL",
        "kernel_mul_mm_iq4_nl_f32",
        "kernel_mul_mm_iq4_nl_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        4,
    );

    // IQ4_XS (136 bytes)
    let weights = gen_iq4_xs(m, k, seed);
    let args = make_args(k, m, n, 256, 136);
    check_correctness(
        "IQ4_XS",
        "kernel_mul_mm_iq4_xs_f32",
        "kernel_mul_mm_iq4_xs_f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights),
        &input_f16,
        &args,
        m,
        n,
        4,
    );

    println!("\nDone.");
}
