// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Alternating A-B benchmark: Standard Q GEMM performance comparison
// Self-written GEMM kernels using same algorithm as llama.cpp

use half::f16;
use infer_train::quant::types::*;
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

fn gen_q4_0(m: usize, k: usize, s: u64) -> Vec<BlockQ4_0> {
    let nb = k / QK;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockQ4_0 { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

fn gen_q8_0(m: usize, k: usize, s: u64) -> Vec<BlockQ8_0> {
    let nb = k / QK;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0i8; 32];
            for j in 0..32 {
                qs[j] = (rng.r#gen::<u8>() as i8).wrapping_sub(127);
            }
            BlockQ8_0 { d: f16::from_f32(1.0).to_bits(), qs }
        })
        .collect()
}

fn gen_q4_1(m: usize, k: usize, s: u64) -> Vec<BlockQ4_1> {
    let nb = k / QK;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockQ4_1 {
                d: f16::from_f32(1.0).to_bits(),
                m: f16::from_f32(0.5).to_bits(),
                qs,
            }
        })
        .collect()
}

fn gen_q5_0(m: usize, k: usize, s: u64) -> Vec<BlockQ5_0> {
    let nb = k / QK;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qh = [0u8; 4];
            let mut qs = [0u8; 16];
            for j in 0..4 {
                qh[j] = rng.r#gen::<u8>();
            }
            for j in 0..16 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockQ5_0 { d: f16::from_f32(1.0).to_bits(), qh, qs }
        })
        .collect()
}

fn gen_q5_1(m: usize, k: usize, s: u64) -> Vec<BlockQ5_1> {
    let nb = k / QK;
    let mut rng = StdRng::seed_from_u64(s);
    (0..m * nb)
        .map(|_| {
            let mut qh = [0u8; 4];
            let mut qs = [0u8; 16];
            for j in 0..4 {
                qh[j] = rng.r#gen::<u8>();
            }
            for j in 0..16 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockQ5_1 {
                d: f16::from_f32(1.0).to_bits(),
                m: f16::from_f32(0.5).to_bits(),
                qh,
                qs,
            }
        })
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

fn run_gemm_test(
    name: &str,
    device: &Device,
    weights_bytes: &[u8],
    input: &[f32],
    m: usize,
    n: usize,
    k: usize,
    block_size: usize,
    rounds: usize,
    gpu_metrics: Option<&gpu_monitor_util::GpuMetrics>,
) -> (f64, f64) {
    println!("\n=== {} GEMM Benchmark ===", name);

    // Load GEMM library (standalone version with standard Q)
    let lib_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");

    let compile_opts = metal::CompileOptions::new();
    let lib = device
        .new_library_with_source(lib_source, &compile_opts)
        .expect("Failed to compile GEMM library");

    // Create pipeline with function constants
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
        "Q4_0" => "kernel_mul_mm_q4_0_f32",
        "Q8_0" => "kernel_mul_mm_q8_0_f32",
        "Q4_1" => "kernel_mul_mm_q4_1_f32",
        "Q5_0" => "kernel_mul_mm_q5_0_f32",
        "Q5_1" => "kernel_mul_mm_q5_1_f32",
        _ => panic!("Unknown format: {}", name),
    };

    let pipeline = device
        .new_compute_pipeline_state_with_function(
            &lib.get_function(kernel_name, Some(cv))
                .expect("Failed to get kernel"),
        )
        .expect("Failed to create pipeline");

    // Create buffers
    let weights_buffer = device.new_buffer_with_data(
        weights_bytes.as_ptr() as *const c_void,
        weights_bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let input_f16: Vec<u16> =
        input.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
    let input_buffer = device.new_buffer_with_data(
        input_f16.as_ptr() as *const c_void,
        (input_f16.len() * 2) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let output_size = m * n * 4;
    let output = device.new_buffer(
        output_size as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // GEMM args
    let nb = k / block_size;
    let args = GemmArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        ne03: 1,
        nb00: weights_bytes.len() as u64 / (m * nb) as u64,
        nb01: (nb * weights_bytes.len() / (m * nb)) as u64,
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

    // Threadgroup size
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

    // Warmup
    for _ in 0..5 {
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
    }

    // Benchmark
    let mut total_us = 0u64;

    for i in 0..rounds {
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

        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let us = start.elapsed().as_micros() as u64;

        total_us += us;

        if (i + 1) % 20 == 0 {
            let avg = total_us as f64 / (i + 1) as f64;
            println!("Run {}/{}: {:.2}us", i + 1, rounds, avg);
        }
    }

    let avg_us = total_us as f64 / rounds as f64;
    println!("  Avg:           {:.2}us", avg_us);

    // Performance/frequency normalization
    let norm = if let Some(gpu) = gpu_metrics {
        let freq = gpu.gpu_freq_mhz;
        println!("  Normalized:    {:.4} us/MHz", avg_us / freq as f64);
        avg_us / freq as f64
    } else {
        0.0
    };

    (avg_us, norm)
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let m = 4096;
    let k = 4096;
    let n = 128;
    let rounds = 100;
    let seed = 42;

    println!("=== Standard Q GEMM Benchmark (N={}) ===", n);
    println!("M={}, K={}, {} rounds\n", m, k, rounds);

    // Get GPU status before benchmark
    let gpu_before = get_gpu_stats();
    if let Some(ref stats) = gpu_before {
        print_stats("Before:", stats);
    }

    let input = gen_f32_matrix(k, n, seed);

    // Test each format
    let tests: Vec<(&str, usize, Vec<u8>)> = vec![
        ("Q4_0", 18, unsafe {
            std::mem::transmute::<Vec<BlockQ4_0>, Vec<u8>>(gen_q4_0(m, k, seed))
        }),
        ("Q8_0", 34, unsafe {
            std::mem::transmute::<Vec<BlockQ8_0>, Vec<u8>>(gen_q8_0(m, k, seed))
        }),
        ("Q4_1", 20, unsafe {
            std::mem::transmute::<Vec<BlockQ4_1>, Vec<u8>>(gen_q4_1(m, k, seed))
        }),
        ("Q5_0", 22, unsafe {
            std::mem::transmute::<Vec<BlockQ5_0>, Vec<u8>>(gen_q5_0(m, k, seed))
        }),
        ("Q5_1", 24, unsafe {
            std::mem::transmute::<Vec<BlockQ5_1>, Vec<u8>>(gen_q5_1(m, k, seed))
        }),
    ];

    let mut all_results = Vec::new();
    for (name, block_size, weights) in tests {
        let (avg_us, norm) = run_gemm_test(
            name,
            &device,
            &weights,
            &input,
            m,
            n,
            k,
            block_size,
            rounds,
            gpu_before.as_ref(),
        );
        all_results.push((name, avg_us, norm));
    }

    println!("\n=== Summary ===");
    println!("{:8} {:>12} {:>12}", "Format", "Avg (us)", "us/MHz");
    println!("{}", "-".repeat(35));
    for (name, avg_us, norm) in all_results {
        println!("{:8} {:>12.2} {:>12.4}", name, avg_us, norm);
    }

    // Get GPU status after benchmark
    let gpu_after = get_gpu_stats();
    if let Some(ref stats) = gpu_after {
        print_stats("After:", stats);
    }

    if let (Some(before), Some(after)) = (&gpu_before, &gpu_after) {
        let diff = before.diff(after);
        print_diff(&diff);

        if diff.freq_mhz.abs() > 50.0 {
            println!("\n⚠ GPU frequency changed {:.0} MHz", diff.freq_mhz);
        } else {
            println!("\n✓ GPU frequency stable");
        }
    }
}
