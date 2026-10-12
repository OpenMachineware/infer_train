// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Flash Attention benchmark: Decode and Prefill

use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 200;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source =
        include_str!("../src/ops/metal/attention/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).unwrap();

    println!("=== Flash Attention Benchmark ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds\n",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );

    let formats = ["f16", "q4_0", "q8_0"];

    // ==================== DECODE ====================
    println!("=== DECODE (seq_q = 1) ===\n");

    for format in &formats {
        println!("Format: {}", format);
        println!(
            "{:>8} {:>6} {:>12} {:>12} {:>10}",
            "seq_k", "NSG", "Orig (µs)", "Adapt (µs)", "Speedup"
        );
        println!("{}", "-".repeat(55));

        for seq_k in [256, 512, 1024, 2048] {
            let result =
                benchmark_decode(&device, &queue, &library, format, seq_k);
            println!(
                "{:>8} {:>6} {:>12.2} {:>12.2} {:>10.2}x",
                seq_k,
                result.nsg,
                result.orig_us,
                result.adapt_us,
                result.speedup
            );
        }
        println!();
    }

    // ==================== PREFILL ====================
    println!("\n=== PREFILL (seq_k = 512) ===\n");

    for format in &formats {
        println!("Format: {}", format);
        println!("{:>8} {:>12}", "seq_q", "Time (µs)");
        println!("{}", "-".repeat(30));

        for seq_q in [16, 32, 64] {
            let time_us = benchmark_prefill(
                &device, &queue, &library, format, seq_q, 512,
            );
            println!("{:>8} {:>12.2}", seq_q, time_us);
        }
        println!();
    }

    // ==================== GFLOPS comparison ====================
    println!("\n=== GFLOPS (F16) ===\n");

    println!("Decode:");
    println!("{:>8} {:>12} {:>12}", "seq_k", "Orig", "Adapt");
    for seq_k in [256, 512, 1024, 2048] {
        let result = benchmark_decode(&device, &queue, &library, "f16", seq_k);
        let orig_gflops =
            2.0 * NUM_HEADS as f64 * seq_k as f64 * HEAD_DIM as f64
                / result.orig_us
                / 1e3;
        let adapt_gflops =
            2.0 * NUM_HEADS as f64 * seq_k as f64 * HEAD_DIM as f64
                / result.adapt_us
                / 1e3;
        println!("{:>8} {:>12.2} {:>12.2}", seq_k, orig_gflops, adapt_gflops);
    }

    println!("\nPrefill (seq_k=512):");
    println!("{:>8} {:>12}", "seq_q", "GFLOPS");
    for seq_q in [16, 32, 64] {
        let time_us =
            benchmark_prefill(&device, &queue, &library, "f16", seq_q, 512);
        let gflops =
            2.0 * NUM_HEADS as f64 * seq_q as f64 * 512.0 * HEAD_DIM as f64
                / time_us
                / 1e3;
        println!("{:>8} {:>12.2}", seq_q, gflops);
    }
}

struct DecodeResult {
    nsg: usize,
    orig_us: f64,
    adapt_us: f64,
    speedup: f64,
}

fn benchmark_decode(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
    seq_k: usize,
) -> DecodeResult {
    let q_data: Vec<u16> = vec![0x3C00; NUM_HEADS * HEAD_DIM];
    let k_data: Vec<u16> = vec![0x3800; seq_k * NUM_HEADS * HEAD_DIM];
    let v_data: Vec<u16> = vec![0x3400; seq_k * NUM_HEADS * HEAD_DIM];

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

    let nsg = if seq_k <= 128 {
        1
    } else if seq_k <= 256 {
        2
    } else if seq_k <= 512 {
        4
    } else {
        8
    };
    let shmem_size = nsg * (2 + HEAD_DIM) * 4;

    let orig_kernel = library
        .get_function(&format!("flash_decode_attention_{}", format), None)
        .unwrap();
    let orig_pipeline =
        device.new_compute_pipeline_state_with_function(&orig_kernel).unwrap();

    let adapt_kernel = library
        .get_function(
            &format!("flash_decode_attention_{}_adaptive", format),
            None,
        )
        .unwrap();
    let adapt_pipeline =
        device.new_compute_pipeline_state_with_function(&adapt_kernel).unwrap();

    let args = build_args(1, seq_k);
    let args_bytes = unsafe {
        std::slice::from_raw_parts(
            &args as *const _ as *const u8,
            std::mem::size_of::<Args>(),
        )
    };

    // Warmup
    for _ in 0..5 {
        run_decode(
            &queue,
            &orig_pipeline,
            args_bytes,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            NUM_HEADS,
            1,
            0,
        );
        run_decode(
            &queue,
            &adapt_pipeline,
            args_bytes,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            NUM_HEADS,
            nsg,
            shmem_size,
        );
    }

    // Alternating benchmark
    let mut orig_times = Vec::with_capacity(NUM_ROUNDS);
    let mut adapt_times = Vec::with_capacity(NUM_ROUNDS);

    for _ in 0..NUM_ROUNDS {
        let t0 = Instant::now();
        run_decode(
            &queue,
            &orig_pipeline,
            args_bytes,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            NUM_HEADS,
            1,
            0,
        );
        orig_times.push(t0.elapsed().as_nanos() as f64);

        let t1 = Instant::now();
        run_decode(
            &queue,
            &adapt_pipeline,
            args_bytes,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            NUM_HEADS,
            nsg,
            shmem_size,
        );
        adapt_times.push(t1.elapsed().as_nanos() as f64);
    }

    orig_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    adapt_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let orig_us = orig_times[NUM_ROUNDS / 2] / 1000.0;
    let adapt_us = adapt_times[NUM_ROUNDS / 2] / 1000.0;

    DecodeResult { nsg, orig_us, adapt_us, speedup: orig_us / adapt_us }
}

fn benchmark_prefill(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
    seq_q: usize,
    seq_k: usize,
) -> f64 {
    let q_data: Vec<u16> = vec![0x3C00; seq_q * NUM_HEADS * HEAD_DIM];
    let k_data: Vec<u16> = vec![0x3800; seq_k * NUM_HEADS * HEAD_DIM];
    let v_data: Vec<u16> = vec![0x3400; seq_k * NUM_HEADS * HEAD_DIM];

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
        (seq_q * NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let kernel = library
        .get_function(&format!("flash_prefill_attention_{}", format), None)
        .unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&kernel).unwrap();

    let args = build_args(seq_q, seq_k);
    let args_bytes = unsafe {
        std::slice::from_raw_parts(
            &args as *const _ as *const u8,
            std::mem::size_of::<Args>(),
        )
    };

    // Warmup
    for _ in 0..5 {
        run_prefill(
            &queue, &pipeline, args_bytes, &q_buf, &k_buf, &v_buf, &out_buf,
            seq_q, NUM_HEADS,
        );
    }

    // Benchmark
    let mut times = Vec::with_capacity(NUM_ROUNDS);

    for _ in 0..NUM_ROUNDS {
        let t0 = Instant::now();
        run_prefill(
            &queue, &pipeline, args_bytes, &q_buf, &k_buf, &v_buf, &out_buf,
            seq_q, NUM_HEADS,
        );
        times.push(t0.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[NUM_ROUNDS / 2] / 1000.0
}

#[repr(C)]
struct Args {
    batch: u32,
    heads: u32,
    seq_q: u32,
    seq_k: u32,
    head_dim: u32,
    scale: f32,
    q_stride_b: u32,
    q_stride_h: u32,
    q_stride_s: u32,
    k_stride_b: u32,
    k_stride_h: u32,
    k_stride_s: u32,
    v_stride_b: u32,
    v_stride_h: u32,
    v_stride_s: u32,
    o_stride_b: u32,
    o_stride_h: u32,
    o_stride_s: u32,
}

fn build_args(seq_q: usize, seq_k: usize) -> Args {
    Args {
        batch: 1,
        heads: NUM_HEADS as u32,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        head_dim: HEAD_DIM as u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        q_stride_b: (NUM_HEADS * seq_q * HEAD_DIM) as u32,
        q_stride_h: (seq_q * HEAD_DIM) as u32,
        q_stride_s: HEAD_DIM as u32,
        k_stride_b: (NUM_HEADS * seq_k * HEAD_DIM * 2) as u32,
        k_stride_h: (seq_k * HEAD_DIM * 2) as u32,
        k_stride_s: (HEAD_DIM * 2) as u32,
        v_stride_b: (NUM_HEADS * seq_k * HEAD_DIM * 2) as u32,
        v_stride_h: (seq_k * HEAD_DIM * 2) as u32,
        v_stride_s: (HEAD_DIM * 2) as u32,
        o_stride_b: (NUM_HEADS * seq_q * HEAD_DIM) as u32,
        o_stride_h: (seq_q * HEAD_DIM) as u32,
        o_stride_s: HEAD_DIM as u32,
    }
}

fn run_decode(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    args_bytes: &[u8],
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    out_buf: &Buffer,
    num_heads: usize,
    nsg: usize,
    shmem_size: usize,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(0, args_bytes.len() as u64, args_bytes.as_ptr() as *const _);
    enc.set_buffer(1, Some(q_buf), 0);
    enc.set_buffer(2, Some(k_buf), 0);
    enc.set_buffer(3, Some(v_buf), 0);
    enc.set_buffer(4, Some(out_buf), 0);
    if shmem_size > 0 {
        enc.set_threadgroup_memory_length(0, shmem_size as u64);
    }
    enc.dispatch_thread_groups(
        MTLSize::new(1, num_heads as u64, 1),
        MTLSize::new(32, nsg as u64, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn run_prefill(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    args_bytes: &[u8],
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    out_buf: &Buffer,
    seq_q: usize,
    num_heads: usize,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(0, args_bytes.len() as u64, args_bytes.as_ptr() as *const _);
    enc.set_buffer(1, Some(q_buf), 0);
    enc.set_buffer(2, Some(k_buf), 0);
    enc.set_buffer(3, Some(v_buf), 0);
    enc.set_buffer(4, Some(out_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(seq_q as u64, num_heads as u64, 1),
        MTLSize::new(128, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}
