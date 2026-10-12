// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Benchmark C=32 vs C=64 for flash decode - All formats

use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 500;

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

    println!("=== Flash Decode C=32 vs C=64 Benchmark ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds\n",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );

    let seq_ks = [256, 512, 1024, 2048];
    let formats = ["f16", "q8_0", "q4_0", "q4_1"];

    println!("| Format | seq_k | C=32 GFLOPS | C=64 GFLOPS | Ratio |");
    println!("|--------|-------|-------------|-------------|-------|");

    for format in &formats {
        for &seq_k in &seq_ks {
            let (gflops_c32, gflops_c64) =
                bench_format(&device, &queue, &library, format, seq_k);
            let ratio = gflops_c64 / gflops_c32 * 100.0;
            println!(
                "| {:6} | {:5} | {:11.2} | {:11.2} | {:5.1}% |",
                format, seq_k, gflops_c32, gflops_c64, ratio
            );
        }
        println!("|--------|-------|-------------|-------------|-------|");
    }
}

fn bench_format(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
    seq_k: usize,
) -> (f64, f64) {
    let (k_buf, v_buf) = create_kv_data(device, format, seq_k);
    let q_buf = create_query_data(device);

    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let partial_buf = device.new_buffer(
        (NUM_HEADS * 16 * 8) as u64,
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

    // Get C=32 kernel
    let kernel_name_c32 =
        format!("flash_decode_multi_wg_partial_{}_nwg8", format);
    let func_c32 = library.get_function(&kernel_name_c32, None).unwrap();
    let pipeline_c32 =
        device.new_compute_pipeline_state_with_function(&func_c32).unwrap();

    // Get C=64 kernel
    let kernel_name_c64 =
        format!("flash_decode_multi_wg_partial_{}_c64_nwg8", format);
    let func_c64 = library.get_function(&kernel_name_c64, None).unwrap();
    let pipeline_c64 =
        device.new_compute_pipeline_state_with_function(&func_c64).unwrap();

    // Get finalize kernel (same for both)
    let finalize_func = library
        .get_function("flash_decode_multi_wg_finalize_f16_nwg8", None)
        .unwrap();
    let finalize_pipeline = device
        .new_compute_pipeline_state_with_function(&finalize_func)
        .unwrap();

    // Benchmark C=32
    let gflops_c32 = run_benchmark(
        queue,
        &pipeline_c32,
        &finalize_pipeline,
        &q_buf,
        &k_buf,
        &v_buf,
        &out_buf,
        &partial_buf,
        &scale_buf,
        &seq_k_buf,
        seq_k,
    );

    // Benchmark C=64
    let gflops_c64 = run_benchmark(
        queue,
        &pipeline_c64,
        &finalize_pipeline,
        &q_buf,
        &k_buf,
        &v_buf,
        &out_buf,
        &partial_buf,
        &scale_buf,
        &seq_k_buf,
        seq_k,
    );

    (gflops_c32, gflops_c64)
}

fn run_benchmark(
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
    seq_k: usize,
) -> f64 {
    const NWG: usize = 8;

    let start = Instant::now();
    for _ in 0..NUM_ROUNDS {
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
        let num_groups = MTLSize::new(NWG as u64, NUM_HEADS as u64, 1);
        encoder.dispatch_thread_groups(num_groups, threads_per_group);
        encoder.end_encoding();

        // Finalize pass
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
    let elapsed = start.elapsed();

    let time_us = elapsed.as_nanos() as f64 / (NUM_ROUNDS as f64 * 1000.0);
    let flops_per_head = 4.0 * seq_k as f64 * HEAD_DIM as f64;
    let total_flops = flops_per_head * NUM_HEADS as f64;
    total_flops / time_us / 1000.0
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
) -> (Buffer, Buffer) {
    let num_elements = NUM_HEADS * seq_k * HEAD_DIM;
    let src_data: Vec<f32> = vec![0.5; num_elements];

    match format {
        "f16" => {
            let data: Vec<u16> = src_data
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            let size = data.len() * 2;
            let buf = device.new_buffer_with_data(
                data.as_ptr() as *const _,
                size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf)
        }
        "q8_0" => {
            let num_blocks = num_elements / 32;
            let mut bytes = vec![0u8; num_blocks * 34]; // Q8_0 block is 34 bytes
            for i in 0..num_blocks {
                let d = half::f16::from_f32(1.0).to_bits();
                bytes[i * 34..i * 34 + 2].copy_from_slice(&d.to_le_bytes());
            }
            let buf = device.new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf)
        }
        "q4_0" => {
            let num_blocks = num_elements / 32;
            let mut bytes = vec![0u8; num_blocks * 18]; // Q4_0 block is 18 bytes
            for i in 0..num_blocks {
                let d = half::f16::from_f32(1.0).to_bits();
                bytes[i * 18..i * 18 + 2].copy_from_slice(&d.to_le_bytes());
            }
            let buf = device.new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf)
        }
        "q4_1" => {
            let num_blocks = num_elements / 32;
            let mut bytes = vec![0u8; num_blocks * 20]; // Q4_1 block is 20 bytes
            for i in 0..num_blocks {
                let d = half::f16::from_f32(1.0).to_bits();
                let m = half::f16::from_f32(0.5).to_bits();
                bytes[i * 20..i * 20 + 2].copy_from_slice(&d.to_le_bytes());
                bytes[i * 20 + 2..i * 20 + 4].copy_from_slice(&m.to_le_bytes());
            }
            let buf = device.new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf)
        }
        _ => panic!("Unknown format: {}", format),
    }
}
