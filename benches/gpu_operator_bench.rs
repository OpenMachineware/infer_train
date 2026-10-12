// GPU Self-written Operator Performance Benchmark
// Tests absolute performance of self-written GPU operators
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use metal::{Device, MTLSize};
use std::time::Instant;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("=== GPU Self-written Operator Performance ===");
    println!("Method: 1000 iterations, measure average time");
    println!();

    test_pool_2d(&device, &queue);
    test_top_k(&device, &queue);
    test_argsort(&device, &queue);
    test_conv_2d(&device, &queue);
}

fn test_pool_2d(device: &Device, queue: &metal::CommandQueue) {
    println!("--- POOL_2D (Max Pooling) ---");

    let source = include_str!("../../shaders/pool_2d.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device
        .new_library_with_source(source, &options)
        .expect("Compile failed");
    let kernel =
        library.get_function("pool_2d_max_f32", None).expect("No kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("No pipeline");

    // Test configs
    let configs = [
        (64, 256, 256, 2, 2, 2, 2),  // Small
        (64, 512, 512, 2, 2, 2, 2),  // Medium
        (128, 256, 256, 3, 3, 2, 2), // Different kernel
    ];

    for (ic, ih, iw, kh, kw, sh, sw) in configs {
        let oh = (ih - kh) / sh + 1;
        let ow = (iw - kw) / sw + 1;
        let input_size = (ic * ih * iw) as usize;
        let output_size = (ic * oh * ow) as usize;

        let input_data: Vec<f32> =
            (0..input_size).map(|i| (i % 1000) as f32 / 100.0).collect();
        let input_buf = device.new_buffer(
            (input_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buf = device.new_buffer(
            (output_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            std::ptr::copy_nonoverlapping(
                input_data.as_ptr(),
                input_buf.contents() as *mut f32,
                input_size,
            );
        }

        // Warmup
        for _ in 0..10 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&input_buf), 0);
            enc.set_buffer(1, Some(&output_buf), 0);
            enc.set_bytes(2, 4, &ic as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(3, 4, &ih as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &iw as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &oh as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &ow as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &kh as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &kw as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &sh as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &sw as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(
                11,
                4,
                &0i32 as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                12,
                4,
                &0i32 as *const i32 as *const std::ffi::c_void,
            );
            enc.dispatch_thread_groups(
                MTLSize { width: output_size as u64, height: 1, depth: 1 },
                MTLSize { width: 64, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Benchmark
        let rounds = 500;
        let mut times = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let start = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&input_buf), 0);
            enc.set_buffer(1, Some(&output_buf), 0);
            enc.set_bytes(2, 4, &ic as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(3, 4, &ih as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &iw as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &oh as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &ow as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &kh as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &kw as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &sh as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &sw as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(
                11,
                4,
                &0i32 as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                12,
                4,
                &0i32 as *const i32 as *const std::ffi::c_void,
            );
            enc.dispatch_thread_groups(
                MTLSize { width: output_size as u64, height: 1, depth: 1 },
                MTLSize { width: 64, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            times.push(start.elapsed().as_nanos() as f64);
        }

        let avg = times.iter().sum::<f64>() / rounds as f64;
        let throughput = (output_size as f64 * 4.0) / (avg * 1e-9) / 1e9; // GB/s
        println!(
            "{}x{}x{} -> {}x{}x{}, {}x{} kernel: {:.2} us, {:.2} GB/s",
            ic,
            ih,
            iw,
            ic,
            oh,
            ow,
            kh,
            kw,
            avg / 1000.0,
            throughput
        );
    }
    println!();
}

fn test_top_k(device: &Device, queue: &metal::CommandQueue) {
    println!("--- TOP_K ---");

    let source = include_str!("../../shaders/top_k.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device
        .new_library_with_source(source, &options)
        .expect("Compile failed");
    let kernel =
        library.get_function("top_k_f32_i32", None).expect("No kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("No pipeline");

    // Test configs: (n, k)
    let configs = [(1024, 8), (4096, 16), (8192, 32)];

    for (n, k) in configs {
        let input_data: Vec<f32> =
            (0..n as usize).map(|i| ((i * 7) % 1000) as f32 / 10.0).collect();
        let input_buf = device.new_buffer(
            (n as usize * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buf = device.new_buffer(
            (k as usize * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            std::ptr::copy_nonoverlapping(
                input_data.as_ptr(),
                input_buf.contents() as *mut f32,
                n as usize,
            );
        }

        // Warmup
        for _ in 0..10 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&input_buf), 0);
            enc.set_buffer(1, Some(&output_buf), 0);
            enc.set_bytes(2, 4, &n as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(3, 4, &k as *const i32 as *const std::ffi::c_void);
            enc.set_threadgroup_memory_length(0, (256 * 4 + 8) as u64);
            enc.set_threadgroup_memory_length(1, 4);
            enc.set_threadgroup_memory_length(2, 4);
            enc.dispatch_thread_groups(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Benchmark
        let rounds = 500;
        let mut times = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let start = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&input_buf), 0);
            enc.set_buffer(1, Some(&output_buf), 0);
            enc.set_bytes(2, 4, &n as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(3, 4, &k as *const i32 as *const std::ffi::c_void);
            enc.set_threadgroup_memory_length(0, (256 * 4 + 8) as u64);
            enc.set_threadgroup_memory_length(1, 4);
            enc.set_threadgroup_memory_length(2, 4);
            enc.dispatch_thread_groups(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            times.push(start.elapsed().as_nanos() as f64);
        }

        let avg = times.iter().sum::<f64>() / rounds as f64;
        println!("n={}, k={}: {:.2} us", n, k, avg / 1000.0);
    }
    println!();
}

fn test_argsort(device: &Device, queue: &metal::CommandQueue) {
    println!("--- ARGSORT (Descending) ---");

    let source = include_str!("../../shaders/argsort.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device
        .new_library_with_source(source, &options)
        .expect("Compile failed");
    let kernel =
        library.get_function("argsort_f32_i32", None).expect("No kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("No pipeline");

    // Test configs
    let configs = [256, 512, 1024];

    for n in configs {
        let input_data: Vec<f32> =
            (0..n as usize).map(|i| ((i * 7) % 1000) as f32 / 10.0).collect();
        let input_buf = device.new_buffer(
            (n as usize * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buf = device.new_buffer(
            (n as usize * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            std::ptr::copy_nonoverlapping(
                input_data.as_ptr(),
                input_buf.contents() as *mut f32,
                n as usize,
            );
        }

        let num_elements: i32 = n;
        let top_k: i32 = n; // Output all sorted indices
        let sort_order: i32 = 1; // Descending

        // Warmup
        for _ in 0..10 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&input_buf), 0);
            enc.set_buffer(1, Some(&output_buf), 0);
            enc.set_bytes(
                2,
                4,
                &num_elements as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                3,
                4,
                &top_k as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                4,
                4,
                &sort_order as *const i32 as *const std::ffi::c_void,
            );
            enc.set_threadgroup_memory_length(0, (n as usize * 4) as u64);
            enc.dispatch_thread_groups(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: n as u64, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Benchmark
        let rounds = 500;
        let mut times = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let start = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&input_buf), 0);
            enc.set_buffer(1, Some(&output_buf), 0);
            enc.set_bytes(
                2,
                4,
                &num_elements as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                3,
                4,
                &top_k as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                4,
                4,
                &sort_order as *const i32 as *const std::ffi::c_void,
            );
            enc.set_threadgroup_memory_length(0, (n as usize * 4) as u64);
            enc.dispatch_thread_groups(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: n as u64, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            times.push(start.elapsed().as_nanos() as f64);
        }

        let avg = times.iter().sum::<f64>() / rounds as f64;
        println!("n={}: {:.2} us", n, avg / 1000.0);
    }
    println!();
}

fn test_conv_2d(device: &Device, queue: &metal::CommandQueue) {
    println!("--- CONV_2D ---");

    let source = include_str!("../../shaders/conv_2d.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device
        .new_library_with_source(source, &options)
        .expect("Compile failed");
    let kernel =
        library.get_function("conv_2d_f32_f32", None).expect("No kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("No pipeline");

    // Test configs: (in_ch, out_ch, in_h, in_w, ker_h, ker_w, stride, pad)
    let configs = [
        (64, 64, 32, 32, 3, 3, 1, 1),
        (64, 128, 32, 32, 3, 3, 1, 1),
        (128, 256, 16, 16, 3, 3, 1, 1),
    ];

    for (in_ch, out_ch, in_h, in_w, ker_h, ker_w, stride, pad) in configs {
        let batch_sz: i32 = 1;
        let out_h = (in_h - ker_h + 2 * pad) / stride + 1;
        let out_w = (in_w - ker_w + 2 * pad) / stride + 1;

        let input_size = (batch_sz * in_ch * in_h * in_w) as usize;
        let weight_size = (out_ch * in_ch * ker_h * ker_w) as usize;
        let output_size = (batch_sz * out_ch * out_h * out_w) as usize;

        let input_data: Vec<f32> = (0..input_size)
            .map(|i| ((i % 1000) as f32 - 500.0) / 100.0)
            .collect();
        let weights: Vec<f32> = (0..weight_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 50.0)
            .collect();

        let input_buf = device.new_buffer(
            (input_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let weight_buf = device.new_buffer(
            (weight_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buf = device.new_buffer(
            (output_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            std::ptr::copy_nonoverlapping(
                input_data.as_ptr(),
                input_buf.contents() as *mut f32,
                input_size,
            );
            std::ptr::copy_nonoverlapping(
                weights.as_ptr(),
                weight_buf.contents() as *mut f32,
                weight_size,
            );
        }

        // Warmup
        for _ in 0..5 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&weight_buf), 0);
            enc.set_buffer(1, Some(&input_buf), 0);
            enc.set_buffer(2, Some(&output_buf), 0);
            enc.set_bytes(
                3,
                4,
                &batch_sz as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                4,
                4,
                &in_ch as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                5,
                4,
                &out_ch as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(6, 4, &in_h as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &in_w as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(
                8,
                4,
                &out_h as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                9,
                4,
                &out_w as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                10,
                4,
                &ker_h as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                11,
                4,
                &ker_w as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                12,
                4,
                &stride as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                13,
                4,
                &stride as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                14,
                4,
                &1i32 as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                15,
                4,
                &1i32 as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(16, 4, &pad as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(17, 4, &pad as *const i32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize { width: output_size as u64, height: 1, depth: 1 },
                MTLSize { width: 64, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Benchmark
        let rounds = 100;
        let mut times = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let start = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&weight_buf), 0);
            enc.set_buffer(1, Some(&input_buf), 0);
            enc.set_buffer(2, Some(&output_buf), 0);
            enc.set_bytes(
                3,
                4,
                &batch_sz as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                4,
                4,
                &in_ch as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                5,
                4,
                &out_ch as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(6, 4, &in_h as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &in_w as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(
                8,
                4,
                &out_h as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                9,
                4,
                &out_w as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                10,
                4,
                &ker_h as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                11,
                4,
                &ker_w as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                12,
                4,
                &stride as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                13,
                4,
                &stride as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                14,
                4,
                &1i32 as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                15,
                4,
                &1i32 as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(16, 4, &pad as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(17, 4, &pad as *const i32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize { width: output_size as u64, height: 1, depth: 1 },
                MTLSize { width: 64, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            times.push(start.elapsed().as_nanos() as f64);
        }

        let avg = times.iter().sum::<f64>() / rounds as f64;
        let flops = output_size as f64 * (in_ch * ker_h * ker_w) as f64 * 2.0; // multiply-add
        let gflops = flops / (avg * 1e-9) / 1e9;
        println!(
            "{}x{}x{}x{} in, {}x{}x{}x{} kernel: {:.2} us, {:.2} GFLOPS",
            batch_sz,
            in_ch,
            in_h,
            in_w,
            out_ch,
            in_ch,
            ker_h,
            ker_w,
            avg / 1000.0,
            gflops
        );
    }
    println!();
}
