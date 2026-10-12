// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Detailed timing analysis for flash decode kernels

use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 100;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let our_source = include_str!("../../shaders/flash_decode_multi_wg.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library =
        device.new_library_with_source(our_source, &options).unwrap();

    println!("=== Flash Decode Detailed Timing Analysis ===");
    println!("Measuring each kernel execution time separately\n");

    let seq_k = 2048u32;
    let nwg = 16;

    // Create buffers
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

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct PartialResult {
        m: f32,
        s: f32,
    }

    let partial_out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * nwg * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let partial_results_buf = device.new_buffer(
        (NUM_HEADS * nwg * std::mem::size_of::<PartialResult>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let seq_k_buf = device.new_buffer_with_data(
        &seq_k as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let scale_buf = device.new_buffer_with_data(
        &(0.125f32) as *const f32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let partial_fn_name =
        format!("flash_decode_multi_wg_partial_f16_nwg{}", nwg);
    let partial_pipeline = device
        .new_compute_pipeline_state_with_function(
            &our_library.get_function(&partial_fn_name, None).unwrap(),
        )
        .unwrap();

    let finalize_fn_name =
        format!("flash_decode_multi_wg_finalize_f16_nwg{}", nwg);
    let finalize_pipeline = device
        .new_compute_pipeline_state_with_function(
            &our_library.get_function(&finalize_fn_name, None).unwrap(),
        )
        .unwrap();

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

    println!("=== Step-by-Step Timing (seq_k={}, NWG={}) ===\n", seq_k, nwg);

    // Measure partial kernel only
    let partial_times: Vec<f64> = (0..NUM_ROUNDS)
        .map(|_| {
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

            let start = Instant::now();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
            let elapsed = start.elapsed();
            elapsed.as_nanos() as f64 / 1000.0
        })
        .collect();

    let partial_avg: f64 =
        partial_times.iter().sum::<f64>() / NUM_ROUNDS as f64;
    let partial_min =
        partial_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let partial_max = partial_times.iter().fold(0.0f64, |a, &b| a.max(b));

    println!("Phase 1 (Partial Kernel):");
    println!("  Average: {:.2} µs", partial_avg);
    println!("  Min:     {:.2} µs", partial_min);
    println!("  Max:     {:.2} µs", partial_max);
    println!();

    // Measure finalize kernel only
    let finalize_times: Vec<f64> = (0..NUM_ROUNDS)
        .map(|_| {
            let cmd_buf = queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&finalize_pipeline);
            encoder.set_buffer(0, Some(&partial_out_buf), 0);
            encoder.set_buffer(1, Some(&partial_results_buf), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);
            encoder.set_buffer(3, Some(&seq_k_buf), 0);

            let threads_per_group = MTLSize::new(32, 1, 1);
            let num_groups = MTLSize::new(1, NUM_HEADS as u64, 1);
            encoder.dispatch_thread_groups(num_groups, threads_per_group);
            encoder.end_encoding();

            let start = Instant::now();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
            let elapsed = start.elapsed();
            elapsed.as_nanos() as f64 / 1000.0
        })
        .collect();

    let finalize_avg: f64 =
        finalize_times.iter().sum::<f64>() / NUM_ROUNDS as f64;
    let finalize_min =
        finalize_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let finalize_max = finalize_times.iter().fold(0.0f64, |a, &b| a.max(b));

    println!("Phase 2 (Finalize Kernel):");
    println!("  Average: {:.2} µs", finalize_avg);
    println!("  Min:     {:.2} µs", finalize_min);
    println!("  Max:     {:.2} µs", finalize_max);
    println!();

    // Measure combined execution
    let combined_times: Vec<f64> = (0..NUM_ROUNDS)
        .map(|_| {
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

            let start = Instant::now();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
            let elapsed = start.elapsed();
            elapsed.as_nanos() as f64 / 1000.0
        })
        .collect();

    let combined_avg: f64 =
        combined_times.iter().sum::<f64>() / NUM_ROUNDS as f64;
    let combined_min =
        combined_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let combined_max = combined_times.iter().fold(0.0f64, |a, &b| a.max(b));

    println!("Combined (Partial + Finalize):");
    println!("  Average: {:.2} µs", combined_avg);
    println!("  Min:     {:.2} µs", combined_min);
    println!("  Max:     {:.2} µs", combined_max);
    println!();

    // Analysis
    println!("=== Performance Breakdown ===");
    println!(
        "Partial:   {:6.2} µs ({:5.1}%)",
        partial_avg,
        100.0 * partial_avg / combined_avg
    );
    println!(
        "Finalize:  {:6.2} µs ({:5.1}%)",
        finalize_avg,
        100.0 * finalize_avg / combined_avg
    );
    println!(
        "Overhead:  {:6.2} µs ({:5.1}%)",
        combined_avg - partial_avg - finalize_avg,
        100.0 * (combined_avg - partial_avg - finalize_avg) / combined_avg
    );
    println!();

    // Compare with single-WG
    let single_source = include_str!("../../shaders/flash_decode_vec.metal");
    let single_library =
        device.new_library_with_source(single_source, &options).unwrap();
    let single_pipeline = device
        .new_compute_pipeline_state_with_function(
            &single_library.get_function("flash_decode_vec_f16", None).unwrap(),
        )
        .unwrap();

    let single_times: Vec<f64> = (0..NUM_ROUNDS)
        .map(|_| {
            let cmd_buf = queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&single_pipeline);
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

            let start = Instant::now();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
            let elapsed = start.elapsed();
            elapsed.as_nanos() as f64 / 1000.0
        })
        .collect();

    let single_avg: f64 = single_times.iter().sum::<f64>() / NUM_ROUNDS as f64;

    println!("=== Comparison ===");
    println!("Single-WG:  {:6.2} µs", single_avg);
    println!("Multi-WG:   {:6.2} µs", combined_avg);
    println!("Speedup:    {:5.2}x", single_avg / combined_avg);
    println!();

    // GFLOPS calculation
    let flops_per_head = 4.0 * seq_k as f64 * HEAD_DIM as f64;
    let total_flops = flops_per_head * NUM_HEADS as f64;

    println!("=== GFLOPS ===");
    println!("Single-WG:  {:6.2} GFLOPS", total_flops / single_avg / 1000.0);
    println!("Multi-WG:   {:6.2} GFLOPS", total_flops / combined_avg / 1000.0);
    println!("  - Partial: {:6.2} GFLOPS", total_flops / partial_avg / 1000.0);
}
