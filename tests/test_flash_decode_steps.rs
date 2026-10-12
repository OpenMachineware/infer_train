// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Step-by-step performance testing
// Measure each step separately and compare with llama.cpp

use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const SEQ_K: u32 = 2048;
const NUM_ROUNDS: usize = 100;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let our_source =
        include_str!("../src/ops/metal/test/test_flash_decode_steps.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library =
        device.new_library_with_source(our_source, &options).unwrap();

    println!("=== Flash Decode Step-by-Step Performance Analysis ===");
    println!(
        "Config: {} heads, {} head_dim, seq_k={}\n",
        NUM_HEADS, HEAD_DIM, SEQ_K
    );

    // Create test data
    let q_data: Vec<f32> = vec![1.0; NUM_HEADS * HEAD_DIM];
    let k_data: Vec<u16> = vec![0x3C00; NUM_HEADS * SEQ_K as usize * HEAD_DIM]; // F16 1.0
    let v_data: Vec<u16> = vec![0x3C00; NUM_HEADS * SEQ_K as usize * HEAD_DIM];

    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        k_data.as_ptr() as *const _,
        (k_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        v_data.as_ptr() as *const _,
        (v_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let seq_k_buf = device.new_buffer_with_data(
        &SEQ_K as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let scale_buf = device.new_buffer_with_data(
        &(0.125f32) as *const f32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Warmup
    println!("Warmup...");
    for _ in 0..20 {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(
            &device
                .new_compute_pipeline_state_with_function(
                    &our_library
                        .get_function("test_query_load_our", None)
                        .unwrap(),
                )
                .unwrap(),
        );
        encoder.set_buffer(0, Some(&q_buf), 0);
        encoder.set_buffer(1, Some(&out_buf), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(32, 1, 1),
        );
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    println!("\n=== Step 1: Query Load ===");
    let query_load_time = benchmark_kernel(
        &device,
        &queue,
        &our_library,
        "test_query_load_our",
        &[&q_buf, &out_buf],
        &[],
        MTLSize::new(1, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    println!("Time: {:.2} µs", query_load_time);
    println!(
        "Bandwidth: {:.2} GB/s",
        (NUM_HEADS * HEAD_DIM * 4) as f64 / query_load_time / 1000.0
    );

    println!("\n=== Step 2: K Read (F16) ===");
    let k_read_time = benchmark_kernel(
        &device,
        &queue,
        &our_library,
        "test_k_read_our",
        &[&k_buf, &out_buf],
        &[&seq_k_buf],
        MTLSize::new(16, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    println!("Time: {:.2} µs", k_read_time);
    println!(
        "Data read: {:.2} MB",
        (NUM_HEADS * SEQ_K as usize * HEAD_DIM * 2) as f64 / 1_000_000.0
    );
    println!(
        "Bandwidth: {:.2} GB/s",
        (NUM_HEADS * SEQ_K as usize * HEAD_DIM * 2) as f64
            / k_read_time
            / 1000.0
    );

    println!("\n=== Step 3: QK Compute ===");
    let qk_compute_time = benchmark_kernel(
        &device,
        &queue,
        &our_library,
        "test_qk_compute_our",
        &[&q_buf, &k_buf, &out_buf],
        &[&seq_k_buf, &scale_buf],
        MTLSize::new(16, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    println!("Time: {:.2} µs", qk_compute_time);
    let qk_flops = 2.0 * NUM_HEADS as f64 * SEQ_K as f64 * HEAD_DIM as f64;
    println!("GFLOPS: {:.2}", qk_flops / qk_compute_time / 1000.0);

    println!("\n=== Step 4: Softmax ===");
    let softmax_time = benchmark_kernel(
        &device,
        &queue,
        &our_library,
        "test_softmax_our",
        &[&out_buf],
        &[],
        MTLSize::new(1, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    println!("Time: {:.2} µs", softmax_time);

    println!("\n=== Step 5: V Accumulation ===");
    let v_accum_time = benchmark_kernel(
        &device,
        &queue,
        &our_library,
        "test_v_accum_our",
        &[&v_buf, &out_buf],
        &[&seq_k_buf],
        MTLSize::new(16, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    println!("Time: {:.2} µs", v_accum_time);
    println!(
        "Data read: {:.2} MB",
        (NUM_HEADS * SEQ_K as usize * HEAD_DIM * 2) as f64 / 1_000_000.0
    );
    println!(
        "Bandwidth: {:.2} GB/s",
        (NUM_HEADS * SEQ_K as usize * HEAD_DIM * 2) as f64
            / v_accum_time
            / 1000.0
    );

    println!("\n=== Step 6: Complete Kernel ===");
    let complete_time = benchmark_kernel(
        &device,
        &queue,
        &our_library,
        "test_complete_our",
        &[&q_buf, &k_buf, &v_buf, &out_buf],
        &[&seq_k_buf, &scale_buf],
        MTLSize::new(16, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    println!("Time: {:.2} µs", complete_time);
    let total_flops = 4.0 * NUM_HEADS as f64 * SEQ_K as f64 * HEAD_DIM as f64;
    println!("GFLOPS: {:.2}", total_flops / complete_time / 1000.0);

    println!("\n=== Performance Breakdown ===");
    println!(
        "Query Load:  {:6.2} µs ({:5.1}%)",
        query_load_time,
        100.0 * query_load_time / complete_time
    );
    println!(
        "K Read:      {:6.2} µs ({:5.1}%)",
        k_read_time,
        100.0 * k_read_time / complete_time
    );
    println!(
        "QK Compute:  {:6.2} µs ({:5.1}%)",
        qk_compute_time,
        100.0 * qk_compute_time / complete_time
    );
    println!(
        "Softmax:     {:6.2} µs ({:5.1}%)",
        softmax_time,
        100.0 * softmax_time / complete_time
    );
    println!(
        "V Accum:     {:6.2} µs ({:5.1}%)",
        v_accum_time,
        100.0 * v_accum_time / complete_time
    );

    println!("\nNote: Individual steps have GPU launch overhead.");
    println!("Combined time is less than sum of individual times.");

    // Now compare with llama.cpp equivalent steps
    println!("\n=== Comparison with llama.cpp ===");
    println!("This requires creating llama.cpp's step-by-step kernels...");
    println!("Coming in next version!");
}

fn benchmark_kernel(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    kernel_name: &str,
    buffers: &[&Buffer],
    constant_buffers: &[&Buffer],
    num_groups: MTLSize,
    threads_per_group: MTLSize,
) -> f64 {
    let pipeline = device
        .new_compute_pipeline_state_with_function(
            &library.get_function(kernel_name, None).unwrap(),
        )
        .unwrap();

    // Warmup
    for _ in 0..10 {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);

        for (i, buf) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(*buf), 0);
        }
        for (i, buf) in constant_buffers.iter().enumerate() {
            encoder.set_buffer((buffers.len() + i) as u64, Some(*buf), 0);
        }

        encoder.dispatch_thread_groups(num_groups, threads_per_group);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    // Benchmark
    let times: Vec<f64> = (0..NUM_ROUNDS)
        .map(|_| {
            let cmd_buf = queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);

            for (i, buf) in buffers.iter().enumerate() {
                encoder.set_buffer(i as u64, Some(*buf), 0);
            }
            for (i, buf) in constant_buffers.iter().enumerate() {
                encoder.set_buffer((buffers.len() + i) as u64, Some(*buf), 0);
            }

            encoder.dispatch_thread_groups(num_groups, threads_per_group);
            encoder.end_encoding();

            let start = Instant::now();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
            start.elapsed().as_nanos() as f64 / 1000.0
        })
        .collect();

    times.iter().sum::<f64>() / NUM_ROUNDS as f64
}
