// Benchmark F32/F16, MXFP4, Q1_0, Q2_0, TQ1_0, TQ2_0 GEMM
// Alternating A-B test against llama.cpp
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use half::f16;
use metal::{Device, MTLSize};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::ffi::c_void;
use std::time::Instant;

mod gpu_monitor_util;
use gpu_monitor_util::get_gpu_stats;

// Block sizes
const QK1_0: usize = 128;
const QK2_0: usize = 64;
const QK4_0: usize = 32;
const QK_MXFP4: usize = 32;
const QK_TQ: usize = 256;

#[repr(C, packed)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlockQ10 {
    d: u16,
    qs: [u8; QK1_0 / 8],
}

#[repr(C, packed)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlockQ20 {
    d: u16,
    qs: [u8; QK2_0 / 4],
}

#[repr(C, packed)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlockMXFP4 {
    e: u8,
    qs: [u8; QK_MXFP4 / 2],
}

#[repr(C, packed)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlockTQ10 {
    d: u16,
    qs: [u8; QK_TQ / 8],
}

#[repr(C, packed)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlockTQ20 {
    d: u16,
    qs: [u8; QK_TQ / 4],
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

fn gen_f16_input(k: usize, n: usize, seed: u64) -> Vec<u16> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..k * n)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 2.0 - 1.0).to_bits())
        .collect()
}

fn bench_format(
    name: &str,
    device: &Device,
    my_lib: &metal::Library,
    llama_lib: &metal::Library,
    weights: &[u8],
    input_f16: &[u16],
    m: usize,
    k: usize,
    n: usize,
    block_bytes: usize,
    nr0: usize,
    rounds: usize,
) {
    let queue = device.new_command_queue();

    const FC_MUL_MM: usize = 700;
    let true_val: bool = true;
    let false_val: bool = false;
    let one_val: i16 = 1;

    let cv = metal::FunctionConstantValues::new();
    unsafe {
        cv.set_constant_value_at_index(
            &true_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            FC_MUL_MM as u64,
        ); // bc_inp
        cv.set_constant_value_at_index(
            &true_val as *const _ as *const c_void,
            metal::MTLDataType::Bool,
            (FC_MUL_MM + 1) as u64,
        ); // bc_out
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

    let kernel_name = format!("kernel_mul_mm_{}_f32", name.to_lowercase());
    let my_fn = my_lib.get_function(&kernel_name, Some(cv.clone())).unwrap();
    let llama_fn = llama_lib.get_function(&kernel_name, Some(cv)).unwrap();

    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_fn).unwrap();
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
    let output_a = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_b = device.new_buffer(
        (m * n * 4) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let args = GemmArgsMm {
        ne00: k as i32,
        ne02: 1,
        nb01: if name == "f32" {
            (k * 4) as u64
        } else if name == "f16" {
            (k * 2) as u64
        } else {
            block_bytes as u64
        },
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

    let nsg = 4;
    let rows_per_tg = nsg * nr0;
    let tg_size = MTLSize::new(32, 4, 1);
    let tg_count =
        MTLSize::new(((m + rows_per_tg - 1) / rows_per_tg) as u64, n as u64, 1);

    // Warmup
    for _ in 0..5 {
        for (pipeline, output) in
            [(&my_pipeline, &output_a), (&llama_pipeline, &output_b)]
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<GemmArgsMm>() as u64,
                &args as *const _ as *const c_void,
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

    // Verify correctness
    let ptr_a = output_a.contents() as *const f32;
    let ptr_b = output_b.contents() as *const f32;
    let slice_a = unsafe { std::slice::from_raw_parts(ptr_a, m * n) };
    let slice_b = unsafe { std::slice::from_raw_parts(ptr_b, m * n) };
    let max_diff = slice_a
        .iter()
        .zip(slice_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Alternating benchmark
    let mut my_total = 0u64;
    let mut llama_total = 0u64;
    let mut my_wins = 0;

    for _ in 0..rounds {
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
        enc.set_buffer(3, Some(&output_a), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let us_a = start.elapsed().as_micros() as u64;
        my_total += us_a;

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
        enc.set_buffer(3, Some(&output_b), 0);
        enc.set_threadgroup_memory_length(0, 8192);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let us_b = start.elapsed().as_micros() as u64;
        llama_total += us_b;

        if us_a < us_b {
            my_wins += 1;
        }
    }

    let my_avg = my_total as f64 / rounds as f64;
    let llama_avg = llama_total as f64 / rounds as f64;
    let ratio = my_avg / llama_avg * 100.0;

    let gpu = get_gpu_stats();
    let freq = gpu.map(|g| g.gpu_freq_mhz as f64).unwrap_or(1.0);
    let my_per_mhz = my_avg / freq;
    let llama_per_mhz = llama_avg / freq;

    println!("  {}:", name);
    println!(
        "    Self:    {:.2} µs ({:.3} µs/MHz @ {:.0} MHz)",
        my_avg, my_per_mhz, freq
    );
    println!("    llama:   {:.2} µs ({:.3} µs/MHz)", llama_avg, llama_per_mhz);
    println!("    Ratio:   {:.1}% ({} / {} wins)", ratio, my_wins, rounds);
    println!(
        "    Max diff: {:.6} {}",
        max_diff,
        if max_diff < 0.01 { "✓" } else { "✗" }
    );
}

fn main() {
    println!(
        "=== GEMM Performance: F32/F16, MXFP4, Q1_0, Q2_0, TQ1_0, TQ2_0 ===\n"
    );

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

    let m = 1024usize;
    let k = 1024usize;
    let n = 1usize;
    let rounds = 100usize;
    let seed = 42u64;

    println!("M={}, K={}, N={}, {} rounds\n", m, k, n, rounds);

    let input_f16 = gen_f16_input(k, n, seed);
    let mut rng = StdRng::seed_from_u64(seed);

    // F32 (raw, no blocks) - NR0=64 in kernel
    println!("--- F32 ---");
    let weights_f32: Vec<f32> =
        (0..m * k).map(|_| rng.r#gen::<f32>() * 0.1).collect();
    bench_format(
        "f32",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights_f32),
        &input_f16,
        m,
        k,
        n,
        4,
        16,
        rounds,
    );

    // F16 (raw, no blocks) - NR0=64 in kernel
    println!("\n--- F16 ---");
    let weights_f16: Vec<u16> = (0..m * k)
        .map(|_| f16::from_f32(rng.r#gen::<f32>() * 0.1).to_bits())
        .collect();
    bench_format(
        "f16",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights_f16),
        &input_f16,
        m,
        k,
        n,
        2,
        16,
        rounds,
    );

    // MXFP4 (32 values/block, 18 bytes)
    println!("\n--- MXFP4 ---");
    let nb_mxfp = k / QK_MXFP4;
    let weights_mxfp: Vec<BlockMXFP4> = (0..m * nb_mxfp)
        .map(|_| {
            let mut qs = [0u8; QK_MXFP4 / 2];
            for j in 0..QK_MXFP4 / 2 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockMXFP4 { e: rng.r#gen::<u8>(), qs }
        })
        .collect();
    bench_format(
        "mxfp4",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights_mxfp),
        &input_f16,
        m,
        k,
        n,
        18,
        2,
        rounds,
    );

    // Q1_0 (128 values/block, 18 bytes, nr0=8)
    println!("\n--- Q1_0 ---");
    let nb_q1 = k / QK1_0;
    let weights_q1: Vec<BlockQ10> = (0..m * nb_q1)
        .map(|_| {
            let mut qs = [0u8; QK1_0 / 8];
            for j in 0..QK1_0 / 8 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockQ10 { d: f16::from_f32(0.1).to_bits(), qs }
        })
        .collect();
    bench_format(
        "q1_0",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights_q1),
        &input_f16,
        m,
        k,
        n,
        18,
        8,
        rounds,
    );

    // Q2_0 (64 values/block, 18 bytes, nr0=4)
    println!("\n--- Q2_0 ---");
    let nb_q2 = k / QK2_0;
    let weights_q2: Vec<BlockQ20> = (0..m * nb_q2)
        .map(|_| {
            let mut qs = [0u8; QK2_0 / 4];
            for j in 0..QK2_0 / 4 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockQ20 { d: f16::from_f32(0.1).to_bits(), qs }
        })
        .collect();
    bench_format(
        "q2_0",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights_q2),
        &input_f16,
        m,
        k,
        n,
        18,
        4,
        rounds,
    );

    // TQ2_0 (256 values/block, 68 bytes)
    // Note: llama.cpp doesn't have TQ1_0, so we only test TQ2_0
    println!("\n--- TQ2_0 ---");
    let nb_tq = k / QK_TQ;
    let weights_tq2: Vec<BlockTQ20> = (0..m * nb_tq)
        .map(|_| {
            let mut qs = [0u8; QK_TQ / 4];
            for j in 0..QK_TQ / 4 {
                qs[j] = rng.r#gen::<u8>();
            }
            BlockTQ20 { d: f16::from_f32(0.1).to_bits(), qs }
        })
        .collect();
    bench_format(
        "tq2_0",
        &device,
        &my_lib,
        &llama_lib,
        bytemuck::cast_slice(&weights_tq2),
        &input_f16,
        m,
        k,
        n,
        68,
        4,
        rounds,
    );

    println!("\n=== Done ===");
}
