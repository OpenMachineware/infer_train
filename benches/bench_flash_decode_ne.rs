// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Benchmark: NE=1 vs NE=4 variants

use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 500;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source = include_str!("../src/ops/metal/flash_decode_vec.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).unwrap();

    println!("=== Flash Decode: NE=1 vs NE=4 ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds\n",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );

    let seq_ks = [256, 512, 1024, 2048, 4096];

    println!(
        "{:>8} {:>12} {:>12} {:>10}",
        "seq_k", "NE=1 (µs)", "NE=4 (µs)", "Winner"
    );
    println!("{}", "-".repeat(50));

    for seq_k in &seq_ks {
        let ne1_result = benchmark_variant(
            &device,
            &queue,
            &library,
            "flash_decode_vec_f16_ne1",
            *seq_k,
        );
        let ne4_result = benchmark_variant(
            &device,
            &queue,
            &library,
            "flash_decode_vec_f16",
            *seq_k,
        );

        let winner = if ne1_result.time_us < ne4_result.time_us {
            format!(
                "NE=1 (+{:.1}%)",
                (ne4_result.time_us / ne1_result.time_us - 1.0) * 100.0
            )
        } else {
            format!(
                "NE=4 (+{:.1}%)",
                (ne1_result.time_us / ne4_result.time_us - 1.0) * 100.0
            )
        };

        println!(
            "{:>8} {:>12.2} {:>12.2} {:>10}",
            seq_k, ne1_result.time_us, ne4_result.time_us, winner
        );
    }
}

struct BenchmarkResult {
    time_us: f64,
}

fn benchmark_variant(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    kernel_name: &str,
    seq_k: usize,
) -> BenchmarkResult {
    // Create data with [heads][seq_k][head_dim] layout
    let q_data: Vec<u16> = vec![0x3C00; NUM_HEADS * HEAD_DIM];
    let k_data: Vec<u16> = vec![0x3800; NUM_HEADS * seq_k * HEAD_DIM];
    let v_data: Vec<u16> = vec![0x3400; NUM_HEADS * seq_k * HEAD_DIM];

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

    let kernel = library.get_function(kernel_name, None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&kernel).unwrap();

    #[repr(C)]
    struct Args {
        seq_length: u32,
        scale: f32,
    }

    let args = Args {
        seq_length: seq_k as u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let args_bytes = unsafe {
        std::slice::from_raw_parts(
            &args as *const _ as *const u8,
            std::mem::size_of::<Args>(),
        )
    };

    let shmem_size = (HEAD_DIM + 32 + HEAD_DIM) * 4;

    // Warmup
    for _ in 0..10 {
        run_kernel(
            &queue, &pipeline, args_bytes, &q_buf, &k_buf, &v_buf, &out_buf,
            NUM_HEADS, shmem_size,
        );
    }

    // Benchmark
    let mut times = Vec::with_capacity(NUM_ROUNDS);
    for _ in 0..NUM_ROUNDS {
        let t0 = Instant::now();
        run_kernel(
            &queue, &pipeline, args_bytes, &q_buf, &k_buf, &v_buf, &out_buf,
            NUM_HEADS, shmem_size,
        );
        times.push(t0.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[NUM_ROUNDS / 2] / 1000.0;

    BenchmarkResult { time_us: median }
}

fn run_kernel(
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
