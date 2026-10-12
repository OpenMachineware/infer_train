// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Benchmark for multi-workgroup flash decode kernel

use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 500;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let our_source =
        include_str!("../src/ops/metal/flash_decode_multi_wg.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library =
        device.new_library_with_source(our_source, &options).unwrap();

    println!("=== Flash Decode Multi-WG: Performance Benchmark ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );
    println!();

    let formats = ["f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"];
    let seq_ks = [256, 512, 1024, 2048];
    let nwg_values = [8, 16, 32];

    for format in &formats {
        println!("\n=== Format: {} ===", format);
        println!(
            "{:>8} {:>6} {:>12} {:>12}",
            "seq_k", "NWG", "Time (µs)", "GFLOPS"
        );
        println!("{}", "-".repeat(50));

        for seq_k in &seq_ks {
            for nwg in &nwg_values {
                if *seq_k as usize / nwg < 1 {
                    continue;
                }

                let result = benchmark_multi_wg(
                    &device,
                    &queue,
                    &our_library,
                    *seq_k as u32,
                    *nwg,
                    format,
                );

                let flops_per_head = 4.0 * *seq_k as f64 * HEAD_DIM as f64;
                let total_flops = flops_per_head * NUM_HEADS as f64;
                let gflops = total_flops / result.time_us / 1000.0;

                println!(
                    "{:>8} {:>6} {:>12.2} {:>11.2}G",
                    seq_k, nwg, result.time_us, gflops
                );
            }
        }
    }

    // Compare with single-workgroup kernel
    println!("\n=== Comparison with Single-WG ===");
    let single_source = include_str!("../src/ops/metal/flash_decode_vec.metal");
    let single_library =
        device.new_library_with_source(single_source, &options).unwrap();

    println!("{:>8} {:>15} {:>15}", "seq_k", "Single (µs)", "Multi-WG (µs)");
    println!("{}", "-".repeat(50));

    for seq_k in &seq_ks {
        let single_result = benchmark_single_wg(
            &device,
            &queue,
            &single_library,
            *seq_k as u32,
        );
        let multi_result = benchmark_multi_wg(
            &device,
            &queue,
            &our_library,
            *seq_k as u32,
            16,
            "f16",
        );

        println!(
            "{:>8} {:>15.2} {:>15.2}",
            seq_k, single_result.time_us, multi_result.time_us
        );
    }
}

struct BenchmarkResult {
    time_us: f64,
}

fn benchmark_multi_wg(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    seq_k: u32,
    nwg: usize,
    format: &str,
) -> BenchmarkResult {
    // Query is always FP32
    let q_data: Vec<f32> = vec![1.0; NUM_HEADS * HEAD_DIM];

    // Generate format-specific KV cache data
    let (k_data, v_data, k_size, v_size) =
        generate_kv_data(format, seq_k as usize);

    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        k_data.as_ptr() as *const _,
        k_size as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        v_data.as_ptr() as *const _,
        v_size as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let partial_out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * nwg * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct PartialResult {
        m: f32,
        s: f32,
    }
    let partial_results_buf = device.new_buffer(
        (NUM_HEADS * nwg * std::mem::size_of::<PartialResult>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let partial_fn_name =
        format!("flash_decode_multi_wg_partial_{}_nwg{}", format, nwg);
    let partial_pipeline = device
        .new_compute_pipeline_state_with_function(
            &library.get_function(&partial_fn_name, None).unwrap(),
        )
        .unwrap();

    let finalize_fn_name =
        format!("flash_decode_multi_wg_finalize_f16_nwg{}", nwg);
    let finalize_pipeline = device
        .new_compute_pipeline_state_with_function(
            &library.get_function(&finalize_fn_name, None).unwrap(),
        )
        .unwrap();

    let seq_k_val = seq_k;
    let scale_val = 0.125f32;

    let seq_k_buf = device.new_buffer_with_data(
        &seq_k_val as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let scale_buf = device.new_buffer_with_data(
        &scale_val as *const f32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    for _ in 0..10 {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&partial_pipeline);
        encoder.set_buffer(0, Some(&seq_k_buf), 0);
        encoder.set_buffer(1, Some(&scale_buf), 0);
        encoder.set_buffer(2, Some(&q_buf), 0);
        encoder.set_buffer(3, Some(&k_buf), 0);
        encoder.set_buffer(4, Some(&v_buf), 0);
        encoder.set_buffer(5, Some(&partial_out_buf), 0);
        encoder.set_buffer(6, Some(&partial_results_buf), 0);

        let threads_per_group = MTLSize::new(32, 1, 1);
        let num_groups = MTLSize::new(nwg as u64, NUM_HEADS as u64, 1);
        encoder.dispatch_thread_groups(num_groups, threads_per_group);
        encoder.end_encoding();

        let encoder2 = cmd_buf.new_compute_command_encoder();
        encoder2.set_compute_pipeline_state(&finalize_pipeline);
        encoder2.set_buffer(0, Some(&partial_out_buf), 0);
        encoder2.set_buffer(1, Some(&partial_results_buf), 0);
        encoder2.set_buffer(2, Some(&out_buf), 0);
        encoder2.set_buffer(3, Some(&seq_k_buf), 0);

        let threads_per_group2 = MTLSize::new(32, 1, 1);
        let num_groups2 = MTLSize::new(1, NUM_HEADS as u64, 1);
        encoder2.dispatch_thread_groups(num_groups2, threads_per_group2);
        encoder2.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    let start = Instant::now();
    for _ in 0..NUM_ROUNDS {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&partial_pipeline);
        encoder.set_buffer(0, Some(&seq_k_buf), 0);
        encoder.set_buffer(1, Some(&scale_buf), 0);
        encoder.set_buffer(2, Some(&q_buf), 0);
        encoder.set_buffer(3, Some(&k_buf), 0);
        encoder.set_buffer(4, Some(&v_buf), 0);
        encoder.set_buffer(5, Some(&partial_out_buf), 0);
        encoder.set_buffer(6, Some(&partial_results_buf), 0);

        let threads_per_group = MTLSize::new(32, 1, 1);
        let num_groups = MTLSize::new(nwg as u64, NUM_HEADS as u64, 1);
        encoder.dispatch_thread_groups(num_groups, threads_per_group);
        encoder.end_encoding();

        let encoder2 = cmd_buf.new_compute_command_encoder();
        encoder2.set_compute_pipeline_state(&finalize_pipeline);
        encoder2.set_buffer(0, Some(&partial_out_buf), 0);
        encoder2.set_buffer(1, Some(&partial_results_buf), 0);
        encoder2.set_buffer(2, Some(&out_buf), 0);
        encoder2.set_buffer(3, Some(&seq_k_buf), 0);

        let threads_per_group2 = MTLSize::new(32, 1, 1);
        let num_groups2 = MTLSize::new(1, NUM_HEADS as u64, 1);
        encoder2.dispatch_thread_groups(num_groups2, threads_per_group2);
        encoder2.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }
    let elapsed = start.elapsed();

    let time_us = elapsed.as_nanos() as f64 / (NUM_ROUNDS as f64 * 1000.0);

    BenchmarkResult { time_us }
}

fn benchmark_single_wg(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    seq_k: u32,
) -> BenchmarkResult {
    let q_data: Vec<u16> = vec![0x3C00; NUM_HEADS * HEAD_DIM];
    let k_data: Vec<u16> = vec![0x3800; NUM_HEADS * seq_k as usize * HEAD_DIM];
    let v_data: Vec<u16> = vec![0x3400; NUM_HEADS * seq_k as usize * HEAD_DIM];

    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 2) as u64,
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

    let pipeline = device
        .new_compute_pipeline_state_with_function(
            &library.get_function("flash_decode_vec_f16", None).unwrap(),
        )
        .unwrap();

    let seq_k_val = seq_k;
    let scale_val = 0.125f32;

    let seq_k_buf = device.new_buffer_with_data(
        &seq_k_val as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let scale_buf = device.new_buffer_with_data(
        &scale_val as *const f32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    for _ in 0..10 {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&seq_k_buf), 0);
        encoder.set_buffer(1, Some(&scale_buf), 0);
        encoder.set_buffer(2, Some(&q_buf), 0);
        encoder.set_buffer(3, Some(&k_buf), 0);
        encoder.set_buffer(4, Some(&v_buf), 0);
        encoder.set_buffer(5, Some(&out_buf), 0);

        let threads_per_group = MTLSize::new(32, 1, 1);
        let num_groups = MTLSize::new(1, NUM_HEADS as u64, 1);
        encoder.dispatch_thread_groups(num_groups, threads_per_group);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    let start = Instant::now();
    for _ in 0..NUM_ROUNDS {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&seq_k_buf), 0);
        encoder.set_buffer(1, Some(&scale_buf), 0);
        encoder.set_buffer(2, Some(&q_buf), 0);
        encoder.set_buffer(3, Some(&k_buf), 0);
        encoder.set_buffer(4, Some(&v_buf), 0);
        encoder.set_buffer(5, Some(&out_buf), 0);

        let threads_per_group = MTLSize::new(32, 1, 1);
        let num_groups = MTLSize::new(1, NUM_HEADS as u64, 1);
        encoder.dispatch_thread_groups(num_groups, threads_per_group);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }
    let elapsed = start.elapsed();

    let time_us = elapsed.as_nanos() as f64 / (NUM_ROUNDS as f64 * 1000.0);

    BenchmarkResult { time_us }
}

// Generate test data for each format
// Returns (k_data, v_data, k_size, v_size)
fn generate_kv_data(
    format: &str,
    seq_k: usize,
) -> (Vec<u8>, Vec<u8>, usize, usize) {
    let num_elements = NUM_HEADS * seq_k * HEAD_DIM;

    match format {
        "f16" => {
            // F16: 2 bytes per element
            let data: Vec<u16> = vec![0x3C00; num_elements];
            let bytes: Vec<u8> =
                data.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let size = bytes.len();
            (bytes.clone(), bytes, size, size)
        }
        "q4_0" => {
            // Q4_0: 18 bytes per 32 elements (2 bytes scale + 16 bytes quantized)
            let block_size = 18;
            let num_blocks = num_elements / 32;
            let data: Vec<u8> = vec![0u8; num_blocks * block_size];
            let size = data.len();
            (data.clone(), data, size, size)
        }
        "q4_1" => {
            // Q4_1: 20 bytes per 32 elements (2 bytes scale + 2 bytes min + 16 bytes quantized)
            let block_size = 20;
            let num_blocks = num_elements / 32;
            let data: Vec<u8> = vec![0u8; num_blocks * block_size];
            let size = data.len();
            (data.clone(), data, size, size)
        }
        "q5_0" => {
            // Q5_0: 22 bytes per 32 elements (2 bytes scale + 4 bytes high bits + 16 bytes quantized)
            let block_size = 22;
            let num_blocks = num_elements / 32;
            let data: Vec<u8> = vec![0u8; num_blocks * block_size];
            let size = data.len();
            (data.clone(), data, size, size)
        }
        "q5_1" => {
            // Q5_1: 24 bytes per 32 elements (2 bytes scale + 2 bytes min + 4 bytes high bits + 16 bytes quantized)
            let block_size = 24;
            let num_blocks = num_elements / 32;
            let data: Vec<u8> = vec![0u8; num_blocks * block_size];
            let size = data.len();
            (data.clone(), data, size, size)
        }
        "q8_0" => {
            // Q8_0: 34 bytes per 32 elements (2 bytes scale + 32 bytes quantized)
            let block_size = 34;
            let num_blocks = num_elements / 32;
            let data: Vec<u8> = vec![0u8; num_blocks * block_size];
            let size = data.len();
            (data.clone(), data, size, size)
        }
        _ => {
            panic!("Unknown format: {}", format);
        }
    }
}
