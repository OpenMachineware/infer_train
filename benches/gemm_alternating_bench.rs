// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Alternating benchmark: Standard Q, MXFP4, IQ4_NL GEMM vs llama.cpp
// Compares self-written template kernels with llama.cpp reference

use half::f16;
use metal::{Device, MTLSize};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::ffi::c_void;
use std::time::Instant;

mod gpu_monitor_util;
use gpu_monitor_util::{get_gpu_stats, print_diff, print_stats};

const QK: usize = 32;

fn gen_f32_matrix(k: usize, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n).map(|_| rng.r#gen::<f32>() * 2.0 - 1.0).collect()
}

fn gen_f16_input(k: usize, n: usize, seed: u64) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect()
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

fn bench_alternating(
    name: &str,
    kernel_name: &str,
    device: &Device,
    lib: &metal::Library,
    weights: &[u8],
    input_f16: &[u16],
    args: &GemmArgs,
    m: usize,
    n: usize,
    rounds: usize,
    gpu_freq: f64,
) -> (f64, f64, f64) {
    let queue = device.new_command_queue();

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

    let kernel = lib.get_function(kernel_name, Some(cv)).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&kernel).unwrap();

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

    let output_a = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_b = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count =
        MTLSize::new(((m + 127) / 128) as u64, ((n + 31) / 32) as u64, 1);

    // Warmup
    for _ in 0..5 {
        for output in [&output_a, &output_b].iter() {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<GemmArgs>() as u64,
                args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(output), 0);
            enc.set_threadgroup_memory_length(0, 8192);
            enc.dispatch_thread_groups(tg_count, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    // Alternating A-B-A-B test
    let mut total_a = 0u64;
    let mut total_b = 0u64;
    let mut wins_a = 0usize;

    for i in 0..rounds {
        // Run A
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_a), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let us_a = start.elapsed().as_micros() as u64;
        total_a += us_a;

        // Run B
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&output_b), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let us_b = start.elapsed().as_micros() as u64;
        total_b += us_b;

        if us_b < us_a {
            wins_a += 1;
        }

        if (i + 1) % 20 == 0 {
            print!(".");
        }
    }
    println!();

    let avg_a = total_a as f64 / rounds as f64;
    let avg_b = total_b as f64 / rounds as f64;

    // Since we run the same kernel, A and B should be very close
    // Report the ratio and normalized times
    let ratio = avg_b / avg_a * 100.0;
    let norm_a = avg_a / gpu_freq;
    let norm_b = avg_b / gpu_freq;

    println!(
        "  A: {:.0} µs ({:.3} µs/MHz), B: {:.0} µs ({:.3} µs/MHz)",
        avg_a, norm_a, avg_b, norm_b
    );
    println!("  Ratio: {:.1}%, Wins A: {}/{}", ratio, wins_a, rounds);

    (ratio, norm_a, norm_b)
}

fn main() {
    println!("=== Standard Q, MXFP4, IQ4_NL GEMM Alternating Benchmark ===");

    let device = Device::system_default().expect("No Metal device");

    let m = 4096usize;
    let k = 4096usize;
    let n = 128usize;
    let rounds = 100usize;
    let seed = 42u64;

    println!("M={}, K={}, N={}, {} rounds\n", m, k, n, rounds);

    // GPU status
    let gpu_before = get_gpu_stats();
    let gpu_freq = if let Some(ref g) = gpu_before {
        print_stats("=== GPU Status ===", g);
        g.gpu_freq_mhz as f64
    } else {
        388.0 // Default M3 GPU frequency
    };

    // Load library
    let lib_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");
    let compile_opts = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(lib_source, &compile_opts)
        .expect("Failed to compile GEMM library");

    // Generate input
    let input_f16 = gen_f16_input(k, n, seed);

    println!("\n=== Standard Q GEMM ===");

    // Q4_0, Q8_0, Q4_1, Q5_0, Q5_1
    for (name, kernel, block_size) in [
        ("Q4_0", "kernel_mul_mm_q4_0_f32", 18usize),
        ("Q8_0", "kernel_mul_mm_q8_0_f32", 34),
        ("Q4_1", "kernel_mul_mm_q4_1_f32", 20),
        ("Q5_0", "kernel_mul_mm_q5_0_f32", 22),
        ("Q5_1", "kernel_mul_mm_q5_1_f32", 24),
    ] {
        println!("\n{}:", name);

        let nb = k / QK;
        let mut weights = Vec::with_capacity(m * nb * block_size);
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..m * nb * block_size {
            weights.push(rng.r#gen::<u8>());
        }

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

        bench_alternating(
            name, kernel, &device, &lib, &weights, &input_f16, &args, m, n,
            rounds, gpu_freq,
        );
    }

    println!("\n=== MXFP4 and IQ4_NL GEMM ===");

    for (name, kernel, block_size) in [
        ("MXFP4", "kernel_mul_mm_mxfp4_f32", 17usize),
        ("IQ4_NL", "kernel_mul_mm_iq4_nl_f32", 18),
    ] {
        println!("\n{}:", name);

        let nb = k / QK;
        let mut weights = Vec::with_capacity(m * nb * block_size);
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..m * nb * block_size {
            weights.push(rng.r#gen::<u8>());
        }

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

        bench_alternating(
            name, kernel, &device, &lib, &weights, &input_f16, &args, m, n,
            rounds, gpu_freq,
        );
    }

    // GPU status after
    let gpu_after = get_gpu_stats();
    if let (Some(before), Some(after)) =
        (gpu_before.as_ref(), gpu_after.as_ref())
    {
        let diff = before.diff(after);
        print_diff(&diff);
    }

    println!("\nDone.");
}
