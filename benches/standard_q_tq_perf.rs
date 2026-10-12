// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Performance test for Standard Q, TQ, and MXFP formats
use metal::*;
use std::ffi::c_void;

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

fn create_cv() -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        let nsg: i16 = 2;
        let ne12: i16 = 1;
        let r2: i16 = 1;
        let r3: i16 = 1;
        cv.set_constant_value_at_index(
            &nsg as *const i16 as *const c_void,
            MTLDataType::Short,
            600,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const i16 as *const c_void,
            MTLDataType::Short,
            602,
        );
        cv.set_constant_value_at_index(
            &r2 as *const i16 as *const c_void,
            MTLDataType::Short,
            603,
        );
        cv.set_constant_value_at_index(
            &r3 as *const i16 as *const c_void,
            MTLDataType::Short,
            604,
        );
    }
    cv
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(true);

    // Load IQ grid tables
    let iq_grid_tables = include_str!("../src/ops/metal/iq_grid_tables.h");

    // Add missing tables for IQ kernels
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

    // Load self-written kernels
    let shader_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let my_source =
        std::fs::read_to_string(shader_dir.join("gemm.metal")).unwrap();
    let my_combined =
        format!("{}\n{}\n{}", iq_extra_tables, iq_grid_tables, my_source);
    let my_lib = device.new_library_with_source(&my_combined, &opts).unwrap();

    let m = 1024usize;
    let k = 4096usize;
    let rounds = 100;

    println!(
        "=== Standard Q + TQ + MXFP Performance (M={}, K={}, {} rounds) ===\n",
        m, k, rounds
    );
    println!("{:12} {:>10}", "Format", "GFLOPS");
    println!("{}", "-".repeat(24));

    // Standard Q formats
    let std_formats: Vec<(&str, usize)> = vec![
        ("Q1_0", 18), // 2 bytes (d) + 16 bytes (qs) = 18 bytes per 128 values
        ("Q4_0", 18), // 2 bytes (d) + 16 bytes (qs) = 18 bytes per 32 values
        ("Q4_1", 20), // 4 bytes (d+m) + 16 bytes (qs) = 20 bytes per 32 values
        ("Q5_0", 22), // 2 bytes (d) + 4 bytes (qh) + 16 bytes (qs) = 22 bytes per 32 values
        ("Q5_1", 24), // 4 bytes (d+m) + 4 bytes (qh) + 16 bytes (qs) = 24 bytes per 32 values
        ("Q8_0", 34), // 2 bytes (d) + 32 bytes (qs) = 34 bytes per 32 values
        ("Q2_0", 18), // 2 bytes (d) + 16 bytes (qs) = 18 bytes per 64 values
    ];

    // TQ and MXFP formats
    let other_formats: Vec<(&str, usize)> = vec![
        ("TQ1_0", 40), // 32 bytes (qs) + 4 bytes (qh) + 4 bytes (scales) + 2 bytes (d) = varies
        ("TQ2_0", 66), // 64 bytes (qs) + 2 bytes (d) = 66 bytes per 256 values
        ("MXFP4", 17), // 1 byte (e) + 16 bytes (qs) = 17 bytes per 32 values
    ];

    let cv = create_cv();

    // Test Standard Q
    println!("\n## Standard Q (7 formats)");
    for (name, block_size) in &std_formats {
        let kernel_name = format!("kernel_mul_mm_{}_f32", name.to_lowercase());
        if let Ok(func) = my_lib.get_function(&kernel_name, Some(cv.clone())) {
            if let Ok(pipeline) =
                device.new_compute_pipeline_state_with_function(&func)
            {
                let gflops = bench_format(
                    &device,
                    &queue,
                    &pipeline,
                    m,
                    k,
                    *block_size,
                    rounds,
                );
                println!("{:12} {:>10.1}", name, gflops);
            }
        } else {
            println!("{:12} {:>10}", name, "N/A");
        }
    }

    // Test TQ + MXFP
    println!("\n## TQ + MXFP (3 formats)");
    for (name, block_size) in &other_formats {
        let kernel_name = format!("kernel_mul_mm_{}_f32", name.to_lowercase());
        if let Ok(func) = my_lib.get_function(&kernel_name, Some(cv.clone())) {
            if let Ok(pipeline) =
                device.new_compute_pipeline_state_with_function(&func)
            {
                let gflops = bench_format(
                    &device,
                    &queue,
                    &pipeline,
                    m,
                    k,
                    *block_size,
                    rounds,
                );
                println!("{:12} {:>10.1}", name, gflops);
            }
        } else {
            println!("{:12} {:>10}", name, "N/A");
        }
    }
}

fn bench_format(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    m: usize,
    k: usize,
    block_size: usize,
    rounds: usize,
) -> f64 {
    let qk = 256; // Standard block size for K-quant
    let nb = k / qk;
    let nb01 = (block_size * nb) as u64;

    let weights_size = m * nb * block_size;
    let weights: Vec<u8> = (0..weights_size).map(|i| (i % 256) as u8).collect();
    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights_size as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let input: Vec<f32> = (0..k).map(|i| (i % 100) as f32 * 0.01).collect();
    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buf = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

    let args = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: block_size as u64,
        nb01,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: (k * 4) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: 1,
        nr0: 2,
        r2: 1,
        r3: 1,
    };

    let tg = MTLSize { width: 32, height: 2, depth: 1 };
    let grid = MTLSize { width: (m as u64 + 3) / 4, height: 1, depth: 1 };

    // Warmup
    for _ in 0..10 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Benchmark
    let mut times = Vec::new();
    for _ in 0..rounds {
        let start = std::time::Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&output_buf), 0);
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        times.push(start.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = times[rounds / 2] / 1000.0;
    let flops = 2.0 * m as f64 * k as f64; // multiply-add = 2 FLOPs per element
    flops / median_us / 1000.0 // GFLOPS
}
