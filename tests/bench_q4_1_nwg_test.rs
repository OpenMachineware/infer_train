// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test different NWG values for Q4_1

use half::f16;
use infer_train::quant::kv_cache::{BlockQ4_1, quantize_row_q4_1};
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

    println!("=== Q4_1 NWG Impact Test ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds\n",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );

    let seq_ks = [256, 512, 1024, 2048];

    println!("| seq_k | NWG=8 (G) | NWG=16 (G) | NWG=32 (G) | Best |");
    println!("|-------|-----------|------------|------------|------|");

    for &seq_k in &seq_ks {
        let g8 = bench_nwg(&device, &queue, &library, seq_k, 8);
        let g16 = bench_nwg(&device, &queue, &library, seq_k, 16);
        let g32 = bench_nwg(&device, &queue, &library, seq_k, 32);

        let best = if g8 >= g16 && g8 >= g32 {
            "NWG=8"
        } else if g16 >= g32 {
            "NWG=16"
        } else {
            "NWG=32"
        };

        println!(
            "| {:5} | {:9.2} | {:10.2} | {:10.2} | {}   |",
            seq_k, g8, g16, g32, best
        );
    }
}

fn bench_nwg(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    seq_k: usize,
    nwg: usize,
) -> f64 {
    let num_elements = NUM_HEADS * seq_k * HEAD_DIM;
    let src_data: Vec<f32> = vec![0.5; num_elements];

    let num_blocks = num_elements / 32;
    let mut blocks: Vec<BlockQ4_1> =
        vec![unsafe { std::mem::zeroed() }; num_blocks];
    quantize_row_q4_1(&src_data, &mut blocks);
    let size = num_blocks * std::mem::size_of::<BlockQ4_1>();
    let k_buf = device.new_buffer_with_data(
        blocks.as_ptr() as *const _,
        size as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = k_buf.clone();

    let q_data: Vec<u16> = vec![0x3C00; NUM_HEADS * HEAD_DIM];
    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

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

    // Get kernel
    let kernel_name =
        format!("flash_decode_multi_wg_partial_q4_1_c64_nwg{}", nwg);
    let func = library.get_function(&kernel_name, None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&func).unwrap();

    // Get finalize kernel
    let finalize_func = library
        .get_function("flash_decode_multi_wg_finalize_f16_nwg8", None)
        .unwrap();
    let finalize_pipeline = device
        .new_compute_pipeline_state_with_function(&finalize_func)
        .unwrap();

    run_benchmark(
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
        seq_k,
        nwg,
    )
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
    nwg: usize,
) -> f64 {
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
        let num_groups = MTLSize::new(nwg as u64, NUM_HEADS as u64, 1);
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
