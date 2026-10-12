// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compare our implementation with llama.cpp style step by step

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
    let llama_source = include_str!(
        "../src/ops/metal/test/test_flash_decode_steps_llama.metal"
    );
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library =
        device.new_library_with_source(our_source, &options).unwrap();
    let llama_library =
        device.new_library_with_source(llama_source, &options).unwrap();

    println!("=== Step-by-Step Comparison: Our vs llama.cpp Style ===");
    println!(
        "Config: {} heads, {} head_dim, seq_k={}\n",
        NUM_HEADS, HEAD_DIM, SEQ_K
    );

    // Create test data
    let q_data: Vec<f32> = vec![1.0; NUM_HEADS * HEAD_DIM];
    let k_data: Vec<u16> = vec![0x3C00; NUM_HEADS * SEQ_K as usize * HEAD_DIM];
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

    println!("\n{:=^70}", "");
    println!("Step 1: Query Load");
    println!("{:=^70}", "");
    let our_query = benchmark_kernel(
        &device,
        &queue,
        &our_library,
        "test_query_load_our",
        &[&q_buf, &out_buf],
        &[],
        MTLSize::new(1, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    let llama_query = benchmark_kernel(
        &device,
        &queue,
        &llama_library,
        "test_query_load_llama_style",
        &[&q_buf, &out_buf],
        &[],
        MTLSize::new(1, 1, 1),
        MTLSize::new(32, 1, 1),
    );
    compare_results("Query Load", our_query, llama_query);

    println!("\n{:=^70}", "");
    println!("Step 2: QK Compute");
    println!("{:=^70}", "");
    let our_qk = benchmark_kernel(
        &device,
        &queue,
        &our_library,
        "test_qk_compute_our",
        &[&q_buf, &k_buf, &out_buf],
        &[&seq_k_buf, &scale_buf],
        MTLSize::new(16, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    let llama_qk = benchmark_kernel(
        &device,
        &queue,
        &llama_library,
        "test_qk_compute_llama_style",
        &[&q_buf, &k_buf, &out_buf],
        &[&seq_k_buf, &scale_buf],
        MTLSize::new(16, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    let qk_flops = 2.0 * NUM_HEADS as f64 * SEQ_K as f64 * HEAD_DIM as f64;
    compare_results_flops("QK Compute", our_qk, llama_qk, qk_flops);

    println!("\n{:=^70}", "");
    println!("Step 3: V Accumulation");
    println!("{:=^70}", "");
    let our_v = benchmark_kernel(
        &device,
        &queue,
        &our_library,
        "test_v_accum_our",
        &[&v_buf, &out_buf],
        &[&seq_k_buf],
        MTLSize::new(16, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    let llama_v = benchmark_kernel(
        &device,
        &queue,
        &llama_library,
        "test_v_accum_llama_style",
        &[&v_buf, &out_buf],
        &[&seq_k_buf],
        MTLSize::new(16, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    compare_results("V Accum", our_v, llama_v);

    println!("\n{:=^70}", "");
    println!("Complete Kernel");
    println!("{:=^70}", "");
    let our_complete = benchmark_kernel(
        &device,
        &queue,
        &our_library,
        "test_complete_our",
        &[&q_buf, &k_buf, &v_buf, &out_buf],
        &[&seq_k_buf, &scale_buf],
        MTLSize::new(16, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    let llama_complete = benchmark_kernel(
        &device,
        &queue,
        &llama_library,
        "test_complete_llama_style",
        &[&q_buf, &k_buf, &v_buf, &out_buf],
        &[&seq_k_buf, &scale_buf],
        MTLSize::new(16, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    let total_flops = 4.0 * NUM_HEADS as f64 * SEQ_K as f64 * HEAD_DIM as f64;
    compare_results_flops(
        "Complete",
        our_complete,
        llama_complete,
        total_flops,
    );

    println!("\n{:=^70}", "");
    println!("Summary");
    println!("{:=^70}", "");
    println!("Total time (us):");
    println!("  Our:     {:.2}", our_complete);
    println!("  llama:   {:.2}", llama_complete);
    println!(
        "  Gap:     {:.1}%",
        (our_complete - llama_complete) / llama_complete * 100.0
    );
    println!("\nConclusion:");
    if our_complete < llama_complete {
        println!(
            "  ✅ Our implementation is FASTER by {:.1}%",
            (llama_complete - our_complete) / llama_complete * 100.0
        );
    } else {
        println!(
            "  ❌ Our implementation is SLOWER by {:.1}%",
            (our_complete - llama_complete) / llama_complete * 100.0
        );
    }
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

fn compare_results(_name: &str, our: f64, llama: f64) {
    println!("  Our:     {:6.2} µs", our);
    println!("  llama:   {:6.2} µs", llama);
    if our < llama {
        println!(
            "  Result:  ✅ FASTER by {:.1}%",
            (llama - our) / llama * 100.0
        );
    } else {
        println!(
            "  Result:  ❌ SLOWER by {:.1}%",
            (our - llama) / llama * 100.0
        );
    }
}

fn compare_results_flops(_name: &str, our: f64, llama: f64, flops: f64) {
    let our_gflops = flops / our / 1000.0;
    let llama_gflops = flops / llama / 1000.0;
    println!("  Our:     {:6.2} µs ({:5.1} GFLOPS)", our, our_gflops);
    println!("  llama:   {:6.2} µs ({:5.1} GFLOPS)", llama, llama_gflops);
    if our < llama {
        println!(
            "  Result:  ✅ FASTER by {:.1}%",
            (llama - our) / llama * 100.0
        );
    } else {
        println!(
            "  Result:  ❌ SLOWER by {:.1}%",
            (our - llama) / llama * 100.0
        );
    }
}
