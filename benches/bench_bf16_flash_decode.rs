// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test BF16 flash decode performance

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

    println!("=== BF16 Flash Decode Performance ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds\n",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );

    let seq_ks = [256, 512, 1024, 2048];

    println!("| seq_k | Best NWG | GFLOPS | llama.cpp | Ratio |");
    println!("|-------|----------|--------|-----------|-------|");

    for &seq_k in &seq_ks {
        let (best_nwg, gflops) =
            find_best_nwg(&device, &queue, &library, seq_k);
        // BF16 should have similar performance to F16
        // F16 llama.cpp baseline: 43.1 GFLOPS at seq_k=2048
        let llama = 43.1 * (seq_k as f64 / 2048.0);
        let ratio = gflops / llama * 100.0;
        let status = if ratio >= 100.0 { "✅" } else { "❌" };
        println!(
            "| {:5} | {:8} | {:6.2} | {:9.2} | {:5.1}% {} |",
            seq_k, best_nwg, gflops, llama, ratio, status
        );
    }
}

fn find_best_nwg(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    seq_k: usize,
) -> (usize, f64) {
    let nwgs = [8, 16, 32];
    let mut best_nwg = 8;
    let mut best_gflops = 0.0;

    for &nwg in &nwgs {
        let kernel_name =
            format!("flash_decode_multi_wg_partial_bf16_c64_nwg{}", nwg);
        if let Ok(func) = library.get_function(&kernel_name, None) {
            let gflops =
                bench_bf16_with_nwg(device, queue, library, seq_k, nwg, &func);
            if gflops > best_gflops {
                best_gflops = gflops;
                best_nwg = nwg;
            }
        }
    }

    (best_nwg, best_gflops)
}

fn bench_bf16_with_nwg(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    seq_k: usize,
    nwg: usize,
    func: &Function,
) -> f64 {
    let num_elements = NUM_HEADS * seq_k * HEAD_DIM;

    // BF16 data: each element is 2 bytes
    // We'll use 0x3F00 (1.0 in BF16) as the test value
    let data: Vec<u16> = vec![0x3F00; num_elements];
    let size = data.len() * 2;
    let k_buf = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        size as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = k_buf.clone();

    let q_data: Vec<u16> = vec![0x3F00; NUM_HEADS * HEAD_DIM];
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

    let pipeline =
        device.new_compute_pipeline_state_with_function(func).unwrap();

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
        let num_groups = MTLSize::new(nwg as u64, NUM_HEADS as u64, 1);
        encoder.dispatch_thread_groups(num_groups, threads_per_group);
        encoder.end_encoding();

        let encoder2 = cmd_buf.new_compute_command_encoder();
        encoder2.set_compute_pipeline_state(&finalize_pipeline);
        encoder2.set_buffer(0, Some(&out_buf), 0);
        encoder2.set_buffer(1, Some(&partial_buf), 0);
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
    total_flops / time_us / 1000.0
}
