// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Test Q4_1 vs Q4_0 with different configurations

use half::f16;
use infer_train::quant::kv_cache::{
    quantize_row_q4_0, quantize_row_q4_1, BlockQ4_0, BlockQ4_1,
};
use metal::*;
use std::time::Instant;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source = include_str!("../src/ops/metal/flash_decode_multi_wg.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).unwrap();

    println!("=== Q4_0 vs Q4_1 Comparison ===");
    println!("Testing with different seq_k values\n");

    let heads_list = [32, 16, 8];
    let seq_ks = [512, 1024, 2048, 4096];

    println!("| Heads | seq_k | Q4_0 (G) | Q4_1 (G) | Ratio Q4_1/Q4_0 |");
    println!("|-------|-------|----------|----------|-----------------|");

    for &num_heads in &heads_list {
        for &seq_k in &seq_ks {
            let q4_0_gflops = bench_format(
                &device, &queue, &library, "q4_0", seq_k, num_heads,
            );
            let q4_1_gflops = bench_format(
                &device, &queue, &library, "q4_1", seq_k, num_heads,
            );
            let ratio = q4_1_gflops / q4_0_gflops * 100.0;
            println!(
                "| {:5} | {:5} | {:8.2} | {:8.2} | {:13.1}% |",
                num_heads, seq_k, q4_0_gflops, q4_1_gflops, ratio
            );
        }
        println!("|-------|-------|----------|----------|-----------------|");
    }
}

fn bench_format(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
    seq_k: usize,
    num_heads: usize,
) -> f64 {
    const HEAD_DIM: usize = 128;
    const NUM_ROUNDS: usize = 300;

    let nwg = if seq_k >= 2048 {
        32
    } else if seq_k >= 1024 {
        16
    } else {
        8
    };

    let num_elements = num_heads * seq_k * HEAD_DIM;
    let src_data: Vec<f32> = vec![0.5; num_elements];

    let (k_buf, v_buf, _size) = match format {
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
            (buf.clone(), buf, size)
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
            (buf.clone(), buf, size)
        }
        _ => panic!("Unknown format"),
    };

    let q_data: Vec<u16> = vec![0x3C00; num_heads * HEAD_DIM];
    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let out_buf = device.new_buffer(
        (num_heads * HEAD_DIM * 4 * nwg) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let partial_buf = device.new_buffer(
        (num_heads * 16 * nwg) as u64,
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

    let kernel_name =
        format!("flash_decode_multi_wg_partial_{}_c64_nwg{}", format, nwg);
    let func = library.get_function(&kernel_name, None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&func).unwrap();

    let finalize_func = library
        .get_function("flash_decode_multi_wg_finalize_f16_nwg8", None)
        .unwrap();
    let finalize_pipeline = device
        .new_compute_pipeline_state_with_function(&finalize_func)
        .unwrap();

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
        encoder.set_buffer(6, Some(&partial_buf), 0);

        let threads_per_group = MTLSize::new(32, 1, 1);
        let num_groups = MTLSize::new(nwg as u64, num_heads as u64, 1);
        encoder.dispatch_thread_groups(num_groups, threads_per_group);
        encoder.end_encoding();

        let encoder2 = cmd_buf.new_compute_command_encoder();
        encoder2.set_compute_pipeline_state(&finalize_pipeline);
        encoder2.set_buffer(0, Some(&out_buf), 0);
        encoder2.set_buffer(1, Some(&partial_buf), 0);
        encoder2.set_buffer(2, Some(&out_buf), 0);
        encoder2.set_buffer(3, Some(&seq_k_buf), 0);

        let threads_per_group2 = MTLSize::new(32, 1, 1);
        let num_groups2 = MTLSize::new(1, num_heads as u64, 1);
        encoder2.dispatch_thread_groups(num_groups2, threads_per_group2);
        encoder2.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }
    let elapsed = start.elapsed();

    let time_us = elapsed.as_nanos() as f64 / (NUM_ROUNDS as f64 * 1000.0);
    let flops_per_head = 4.0 * seq_k as f64 * HEAD_DIM as f64;
    let total_flops = flops_per_head * num_heads as f64;
    total_flops / time_us / 1000.0
}
