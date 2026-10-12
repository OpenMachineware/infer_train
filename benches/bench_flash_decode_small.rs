// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Test flash decode performance for small seq_k (1-256)

use half::f16;
use infer_train::quant::kv_cache::{
    quantize_row_q4_0, quantize_row_q4_1, quantize_row_q5_0, quantize_row_q5_1,
    quantize_row_q8_0, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0,
};
use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 1000;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source =
        include_str!("../src/ops/metal/attention/flash_decode_multi_wg.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = match device.new_library_with_source(source, &options) {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("Metal compilation error: {}", e);
            std::process::exit(1);
        }
    };

    println!("=== Flash Decode Small seq_k Performance ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds\n",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );

    let formats = ["f16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1"];
    let seq_ks = [1, 8, 16, 32, 64, 128, 256];

    for format in &formats {
        println!("\n=== Format: {} ===", format);
        println!(
            "{:>8} {:>10} {:>12} {:>10}",
            "seq_k", "Time (µs)", "GFLOPS", "GB/s"
        );

        for &seq_k in &seq_ks {
            let (time_us, gflops, gbps) =
                bench_format(&device, &queue, &library, format, seq_k);
            println!(
                "{:>8} {:>10.2} {:>12.2} {:>10.2}",
                seq_k, time_us, gflops, gbps
            );
        }
    }

    println!("\n=== Note ===");
    println!("Small seq_k may have overhead dominance (kernel launch, synchronization)");
    println!("GFLOPS is very low for small seq_k due to fixed overhead");
}

fn bench_format(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
    seq_k: usize,
) -> (f64, f64, f64) {
    // For small seq_k, use NWG=8
    let nwg = 8;

    let (k_buf, v_buf, block_size) = create_kv_data(device, format, seq_k);
    let q_buf = create_query_data(device);

    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4 * nwg) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let partial_buf = device.new_buffer(
        (NUM_HEADS * 16 * nwg) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
    let scale_buf = device.new_buffer_with_data(
        &scale as *const f32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let seq_k_buf = device.new_buffer_with_data(
        &(seq_k as u32) as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Use C=64 kernel for all formats
    let kernel_name =
        format!("flash_decode_multi_wg_partial_{}_c64_nwg8", format);
    let func = library.get_function(&kernel_name, None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&func).unwrap();

    let finalize_func = library
        .get_function("flash_decode_multi_wg_finalize_f16_nwg8", None)
        .unwrap();
    let finalize_pipeline = device
        .new_compute_pipeline_state_with_function(&finalize_func)
        .unwrap();

    // Warmup
    for _ in 0..100 {
        run_kernel(
            queue,
            &pipeline,
            &finalize_pipeline,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            &partial_buf,
            &scale_buf,
            &seq_k_buf,
            nwg,
        );
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..NUM_ROUNDS {
        run_kernel(
            queue,
            &pipeline,
            &finalize_pipeline,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            &partial_buf,
            &scale_buf,
            &seq_k_buf,
            nwg,
        );
    }
    let elapsed = start.elapsed();

    let time_us = elapsed.as_nanos() as f64 / (NUM_ROUNDS as f64 * 1000.0);
    let flops_per_head = 4.0 * seq_k as f64 * HEAD_DIM as f64;
    let total_flops = flops_per_head * NUM_HEADS as f64;
    let gflops = total_flops / time_us / 1000.0;

    // Bandwidth (K + V reads + output write)
    let bytes_transferred = match format {
        "f16" => {
            seq_k * NUM_HEADS * HEAD_DIM * 2 * 2 + NUM_HEADS * HEAD_DIM * 4
        } // K + V (2 bytes each) + output
        "q8_0" => {
            seq_k * NUM_HEADS * (HEAD_DIM / 32) * 34 * 2
                + NUM_HEADS * HEAD_DIM * 4
        }
        "q5_0" => {
            seq_k * NUM_HEADS * (HEAD_DIM / 32) * 22 * 2
                + NUM_HEADS * HEAD_DIM * 4
        }
        "q5_1" => {
            seq_k * NUM_HEADS * (HEAD_DIM / 32) * 24 * 2
                + NUM_HEADS * HEAD_DIM * 4
        }
        "q4_0" => {
            seq_k * NUM_HEADS * (HEAD_DIM / 32) * 18 * 2
                + NUM_HEADS * HEAD_DIM * 4
        }
        "q4_1" => {
            seq_k * NUM_HEADS * (HEAD_DIM / 32) * 20 * 2
                + NUM_HEADS * HEAD_DIM * 4
        }
        _ => 0,
    };
    let gbps = bytes_transferred as f64 / time_us / 1000.0;

    (time_us, gflops, gbps)
}

fn run_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    finalize_pipeline: &ComputePipelineState,
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    out_buf: &Buffer,
    partial_buf: &Buffer,
    scale_buf: &Buffer,
    seq_k_buf: &Buffer,
    nwg: usize,
) {
    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(seq_k_buf), 0);
    encoder.set_buffer(1, Some(scale_buf), 0);
    encoder.set_buffer(2, Some(q_buf), 0);
    encoder.set_buffer(3, Some(k_buf), 0);
    encoder.set_buffer(4, Some(v_buf), 0);
    encoder.set_buffer(5, Some(out_buf), 0);
    encoder.set_buffer(6, Some(partial_buf), 0);

    let threads_per_group = MTLSize::new(32, 1, 1);
    let num_groups = MTLSize::new(nwg as u64, NUM_HEADS as u64, 1);
    encoder.dispatch_thread_groups(num_groups, threads_per_group);
    encoder.end_encoding();

    let encoder2 = cmd_buf.new_compute_command_encoder();
    encoder2.set_compute_pipeline_state(finalize_pipeline);
    encoder2.set_buffer(0, Some(out_buf), 0);
    encoder2.set_buffer(1, Some(partial_buf), 0);
    encoder2.set_buffer(2, Some(out_buf), 0);
    encoder2.set_buffer(3, Some(seq_k_buf), 0);

    let threads_per_group2 = MTLSize::new(32, 1, 1);
    let num_groups2 = MTLSize::new(1, NUM_HEADS as u64, 1);
    encoder2.dispatch_thread_groups(num_groups2, threads_per_group2);
    encoder2.end_encoding();

    cmd_buf.commit();
    cmd_buf.wait_until_completed();
}

fn create_query_data(device: &Device) -> Buffer {
    let q_data: Vec<u16> = vec![0x3C00; NUM_HEADS * HEAD_DIM];
    device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn create_kv_data(
    device: &Device,
    format: &str,
    seq_k: usize,
) -> (Buffer, Buffer, usize) {
    let num_elements = NUM_HEADS * seq_k * HEAD_DIM;
    let src_data: Vec<f32> = vec![0.5; num_elements];

    match format {
        "f16" => {
            let data: Vec<u16> =
                src_data.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
            let size = data.len() * 2;
            let buf = device.new_buffer_with_data(
                data.as_ptr() as *const _,
                size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, 2)
        }
        "q8_0" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ8_0> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q8_0(&src_data, &mut blocks);
            let size = num_blocks * 34;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, 34)
        }
        "q5_0" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ5_0> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q5_0(&src_data, &mut blocks);
            let size = num_blocks * 22;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, 22)
        }
        "q5_1" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ5_1> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q5_1(&src_data, &mut blocks);
            let size = num_blocks * 24;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, 24)
        }
        "q4_0" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ4_0> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q4_0(&src_data, &mut blocks);
            let size = num_blocks * 18;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, 18)
        }
        "q4_1" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ4_1> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q4_1(&src_data, &mut blocks);
            let size = num_blocks * 20;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, 20)
        }
        _ => panic!("Unknown format: {}", format),
    }
}
