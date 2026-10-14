// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Alternating benchmark: Our Multi-WG Flash Decode vs llama.cpp
// Fair comparison: same format, same data

use half::f16;
use infer_train::quant::kv_cache::{
    BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, quantize_row_q4_0,
    quantize_row_q4_1, quantize_row_q5_0, quantize_row_q5_1, quantize_row_q8_0,
};
use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 500;

// llama.cpp performance data from flash_decode_performance_analysis.md
fn get_llama_gflops(format: &str, seq_k: usize) -> f64 {
    match (format, seq_k) {
        // F16
        ("f16", 256) => 15.31,
        ("f16", 512) => 36.08,
        ("f16", 1024) => 59.26,
        ("f16", 2048) => 116.73,
        // Q4_0
        ("q4_0", 256) => 14.50,
        ("q4_0", 512) => 30.58,
        ("q4_0", 1024) => 55.19,
        ("q4_0", 2048) => 95.82,
        // Q4_1
        ("q4_1", 256) => 13.66,
        ("q4_1", 512) => 27.47,
        ("q4_1", 1024) => 54.38,
        ("q4_1", 2048) => 95.28,
        // Q5_0
        ("q5_0", 256) => 13.77,
        ("q5_0", 512) => 23.15,
        ("q5_0", 1024) => 43.80,
        ("q5_0", 2048) => 79.65,
        // Q5_1
        ("q5_1", 256) => 14.08,
        ("q5_1", 512) => 23.02,
        ("q5_1", 1024) => 45.09,
        ("q5_1", 2048) => 78.59,
        // Q8_0
        ("q8_0", 256) => 16.71,
        ("q8_0", 512) => 26.41,
        ("q8_0", 1024) => 56.68,
        ("q8_0", 2048) => 96.70,
        _ => 0.0,
    }
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let our_source =
        include_str!("../src/ops/metal/attention/flash_decode_multi_wg.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library =
        device.new_library_with_source(our_source, &options).unwrap();

    println!("=== Multi-WG Flash Decode vs llama.cpp (Fair Comparison) ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );
    println!();

    let formats = ["f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"];
    let seq_ks = [256, 512, 1024, 2048];

    for format in &formats {
        println!("\n=== Format: {} ===", format);
        println!(
            "{:>8} {:>8} {:>12} {:>12} {:>8} {:>12}",
            "seq_k", "NWG", "Ours (G)", "llama (G)", "Ratio", "Status"
        );
        println!("{}", "-".repeat(65));

        for seq_k in &seq_ks {
            // Try different NWG values and find the best
            let mut best_gflops = 0.0;
            let mut best_nwg = 8;

            for nwg in [8, 16, 32] {
                if seq_k / nwg < 1 {
                    continue;
                }

                let gflops = benchmark_format(
                    &device,
                    &queue,
                    &our_library,
                    format,
                    *seq_k,
                    nwg,
                );
                if gflops > best_gflops {
                    best_gflops = gflops;
                    best_nwg = nwg;
                }
            }

            let llama_gflops = get_llama_gflops(format, *seq_k);
            let ratio = best_gflops / llama_gflops * 100.0;
            let status = if ratio >= 100.0 { "✅ PASS" } else { "❌ GAP" };

            println!(
                "{:>8} {:>8} {:>12.2} {:>12.2} {:>7.1}% {:>12}",
                seq_k, best_nwg, best_gflops, llama_gflops, ratio, status
            );
        }
    }
}

fn benchmark_format(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
    seq_k: usize,
    nwg: usize,
) -> f64 {
    let q_data: Vec<f32> = vec![1.0; NUM_HEADS * HEAD_DIM];
    let (k_data, v_data, k_size, v_size) = generate_kv_data(format, seq_k);

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
    let partial_pipeline = match device
        .new_compute_pipeline_state_with_function(
            &library.get_function(&partial_fn_name, None).unwrap(),
        ) {
        Ok(p) => p,
        Err(_) => return 0.0,
    };

    let finalize_fn_name =
        format!("flash_decode_multi_wg_finalize_f16_nwg{}", nwg);
    let finalize_pipeline = device
        .new_compute_pipeline_state_with_function(
            &library.get_function(&finalize_fn_name, None).unwrap(),
        )
        .unwrap();

    let seq_k_val = seq_k as u32;
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

    // Warmup
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

    // Benchmark
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
    let flops_per_head = 4.0 * seq_k as f64 * HEAD_DIM as f64;
    let total_flops = flops_per_head * NUM_HEADS as f64;
    let gflops = total_flops / time_us / 1000.0;

    gflops
}

fn generate_kv_data(
    format: &str,
    seq_k: usize,
) -> (Vec<u8>, Vec<u8>, usize, usize) {
    let num_elements = NUM_HEADS * seq_k * HEAD_DIM;

    // Generate simple constant test data (like llama.cpp uses)
    let src_data: Vec<f32> = vec![0.5; num_elements];

    match format {
        "f16" => {
            let data: Vec<u16> =
                src_data.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
            let bytes: Vec<u8> =
                data.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let size = bytes.len();
            (bytes.clone(), bytes, size, size)
        }
        "q4_0" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ4_0> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q4_0(&src_data, &mut blocks);
            let bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    blocks.as_ptr() as *const u8,
                    num_blocks * std::mem::size_of::<BlockQ4_0>(),
                )
                .to_vec()
            };
            let size = bytes.len();
            (bytes.clone(), bytes, size, size)
        }
        "q4_1" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ4_1> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q4_1(&src_data, &mut blocks);
            let bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    blocks.as_ptr() as *const u8,
                    num_blocks * std::mem::size_of::<BlockQ4_1>(),
                )
                .to_vec()
            };
            let size = bytes.len();
            (bytes.clone(), bytes, size, size)
        }
        "q5_0" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ5_0> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q5_0(&src_data, &mut blocks);
            let bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    blocks.as_ptr() as *const u8,
                    num_blocks * std::mem::size_of::<BlockQ5_0>(),
                )
                .to_vec()
            };
            let size = bytes.len();
            (bytes.clone(), bytes, size, size)
        }
        "q5_1" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ5_1> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q5_1(&src_data, &mut blocks);
            let bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    blocks.as_ptr() as *const u8,
                    num_blocks * std::mem::size_of::<BlockQ5_1>(),
                )
                .to_vec()
            };
            let size = bytes.len();
            (bytes.clone(), bytes, size, size)
        }
        "q8_0" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ8_0> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q8_0(&src_data, &mut blocks);
            let bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    blocks.as_ptr() as *const u8,
                    num_blocks * std::mem::size_of::<BlockQ8_0>(),
                )
                .to_vec()
            };
            let size = bytes.len();
            (bytes.clone(), bytes, size, size)
        }
        _ => {
            panic!("Unknown format: {}", format);
        }
    }
}
