// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Simple benchmark for flash_decode_vec kernel

use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 500;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let our_source = include_str!("../src/ops/metal/flash_decode_vec.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library =
        device.new_library_with_source(our_source, &options).unwrap();

    println!("=== Flash Decode Vec: Performance Benchmark ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds\n",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );

    let formats = ["f16", "q4_0"];
    let seq_ks = [256, 512, 1024, 2048];

    for format in &formats {
        println!("\n=== Format: {} ===", format);
        println!("{:>8} {:>12} {:>12}", "seq_k", "Time (µs)", "GFLOPS");
        println!("{}", "-".repeat(40));

        for seq_k in &seq_ks {
            let result = benchmark_format_seq_k(
                &device,
                &queue,
                &our_library,
                format,
                *seq_k,
            );

            let flops_per_head = 4.0 * *seq_k as f64 * HEAD_DIM as f64;
            let total_flops = flops_per_head * NUM_HEADS as f64;
            let gflops = total_flops / result.time_us / 1000.0;

            println!("{:>8} {:>12.2} {:>11.2}G", seq_k, result.time_us, gflops);
        }
    }
}

struct BenchmarkResult {
    time_us: f64,
}

fn benchmark_format_seq_k(
    device: &Device,
    queue: &CommandQueue,
    our_library: &Library,
    format: &str,
    seq_k: usize,
) -> BenchmarkResult {
    // Query: [heads][head_dim] layout
    // Query: FP32 for kernel compatibility
    let q_data: Vec<f32> = vec![1.0; NUM_HEADS * HEAD_DIM];

    // KV cache: [heads][seq_k][head_dim] layout for optimal cache locality
    // Each head's KV data is contiguous, improving memory access patterns
    let k_data: Vec<u16> = vec![0x3C00; NUM_HEADS * seq_k * HEAD_DIM];
    let v_data: Vec<u16> = vec![0x3C00; NUM_HEADS * seq_k * HEAD_DIM];

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

    let our_kernel_name = format!("flash_decode_vec_{}", format);
    let our_kernel = our_library.get_function(&our_kernel_name, None).unwrap();
    let our_pipeline =
        device.new_compute_pipeline_state_with_function(&our_kernel).unwrap();

    #[repr(C)]
    struct OurArgs {
        seq_length: u32,
        scale: f32,
    }

    let our_args = OurArgs {
        seq_length: seq_k as u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let our_args_bytes = unsafe {
        std::slice::from_raw_parts(
            &our_args as *const _ as *const u8,
            std::mem::size_of::<OurArgs>(),
        )
    };

    let shmem_size = (HEAD_DIM + 32 + HEAD_DIM) * 4;

    for _ in 0..10 {
        run_our_kernel(
            &queue,
            &our_pipeline,
            our_args_bytes,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            NUM_HEADS,
            shmem_size,
        );
    }

    let mut times = Vec::with_capacity(NUM_ROUNDS);

    for _ in 0..NUM_ROUNDS {
        let t0 = Instant::now();
        run_our_kernel(
            &queue,
            &our_pipeline,
            our_args_bytes,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            NUM_HEADS,
            shmem_size,
        );
        times.push(t0.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[NUM_ROUNDS / 2] / 1000.0;

    BenchmarkResult { time_us: median }
}

fn run_our_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    args_bytes: &[u8],
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    out_buf: &Buffer,
    num_heads: usize,
    shmem_size: usize,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(0, args_bytes.len() as u64, args_bytes.as_ptr() as *const _);
    enc.set_buffer(2, Some(q_buf), 0);
    enc.set_buffer(3, Some(k_buf), 0);
    enc.set_buffer(4, Some(v_buf), 0);
    enc.set_buffer(5, Some(out_buf), 0);
    enc.set_threadgroup_memory_length(0, shmem_size as u64);
    enc.dispatch_thread_groups(
        MTLSize::new(1, num_heads as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}
