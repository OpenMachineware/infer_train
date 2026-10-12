// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Comprehensive flash attention decode benchmark: all formats, all sizes

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

    println!("=== Flash Attention Decode: All Formats ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds\n",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );

    let seq_ks = [256, 512, 1024, 2048, 4096];

    // Test all formats
    let formats: Vec<(&str, &str)> = vec![
        ("f16", "flash_decode_attention_f16"),
        ("bf16", "flash_decode_attention_bf16"),
        ("q4_0", "flash_decode_attention_q4_0"),
        ("q4_1", "flash_decode_attention_q4_1"),
        ("q5_0", "flash_decode_attention_q5_0"),
        ("q5_1", "flash_decode_attention_q5_1"),
        ("q8_0", "flash_decode_attention_q8_0"),
        ("iq4_nl", "flash_decode_attention_iq4_nl"),
        ("q2_K", "flash_decode_attention_q2_K"),
        ("q3_K", "flash_decode_attention_q3_K"),
        ("q4_K", "flash_decode_attention_q4_K"),
        ("q5_K", "flash_decode_attention_q5_K"),
        ("q6_K", "flash_decode_attention_q6_K"),
    ];

    for (fmt_name, kernel_name) in &formats {
        println!("\n=== Format: {} ===", fmt_name);
        println!(
            "{:>8} {:>6} {:>12} {:>8}",
            "seq_k", "NSG", "Time (µs)", "GFLOPS"
        );
        println!("{}", "-".repeat(42));

        for seq_k in &seq_ks {
            let result = benchmark_format_seq_k(
                &device,
                &queue,
                &library,
                fmt_name,
                kernel_name,
                *seq_k,
            );

            let gflops = compute_gflops(*seq_k, result.time_us);

            println!(
                "{:>8} {:>6} {:>12.2} {:>8.2}",
                seq_k, result.nsg, result.time_us, gflops
            );
        }
    }
}

struct BenchmarkResult {
    nsg: usize,
    time_us: f64,
}

fn compute_gflops(seq_k: usize, time_us: f64) -> f64 {
    // Flash attention FLOPs: 4 * seq_k * head_dim (2 matmul + softmax + output)
    let flops = 4.0 * (seq_k as f64) * (HEAD_DIM as f64) * (NUM_HEADS as f64);
    let time_s = time_us / 1_000_000.0;
    flops / time_s / 1_000_000_000.0
}

fn benchmark_format_seq_k(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    fmt_name: &str,
    kernel_name: &str,
    seq_k: usize,
) -> BenchmarkResult {
    // Determine block size for format
    let block_size = get_block_size(fmt_name);

    // Create test data based on format
    let (q_data, k_data, v_data) =
        create_test_data(fmt_name, seq_k, block_size);

    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        k_data.as_ptr() as *const _,
        (k_data.len()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        v_data.as_ptr() as *const _,
        (v_data.len()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Compute NSG based on seq_k (same logic as llama.cpp)
    let ncpsg = 32; // cache items per SIMD group
    let mut nsg = 1;
    while 2 * 32 * nsg * ncpsg < seq_k && nsg < 4 {
        nsg *= 2;
    }

    // For adaptive kernel, use dynamic NSG
    let nsg = compute_adaptive_nsg(seq_k);

    // Args struct
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

    let k_stride_s = block_size as u32;
    let k_stride_h = (seq_k * block_size) as u32;
    let k_stride_b = (NUM_HEADS * seq_k * block_size) as u32;

    let args = Args {
        batch: 1,
        heads: NUM_HEADS as u32,
        seq_q: 1,
        seq_k: seq_k as u32,
        head_dim: HEAD_DIM as u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        q_stride_b: (NUM_HEADS * HEAD_DIM * 2) as u32,
        q_stride_h: (HEAD_DIM * 2) as u32,
        q_stride_s: (HEAD_DIM * 2) as u32,
        k_stride_b: k_stride_b,
        k_stride_h: k_stride_h,
        k_stride_s: k_stride_s,
        v_stride_b: k_stride_b,
        v_stride_h: k_stride_h,
        v_stride_s: k_stride_s,
        o_stride_b: (NUM_HEADS * HEAD_DIM * 4) as u32,
        o_stride_h: (HEAD_DIM * 4) as u32,
        o_stride_s: (HEAD_DIM * 4) as u32,
    };

    let args_bytes = unsafe {
        std::slice::from_raw_parts(
            &args as *const _ as *const u8,
            std::mem::size_of::<Args>(),
        )
    };

    // Get kernel
    let kernel = library.get_function(kernel_name, None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&kernel).unwrap();

    // Threadgroup memory size
    let shmem_size = nsg * (2 + HEAD_DIM) * 4;

    // Warmup
    for _ in 0..5 {
        run_kernel(
            &queue, &pipeline, args_bytes, &q_buf, &k_buf, &v_buf, &out_buf,
            NUM_HEADS, nsg, shmem_size,
        );
    }

    // Benchmark
    let mut times = Vec::with_capacity(NUM_ROUNDS);

    for _ in 0..NUM_ROUNDS {
        let t0 = Instant::now();
        run_kernel(
            &queue, &pipeline, args_bytes, &q_buf, &k_buf, &v_buf, &out_buf,
            NUM_HEADS, nsg, shmem_size,
        );
        times.push(t0.elapsed().as_nanos() as f64);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[NUM_ROUNDS / 2] / 1000.0; // µs

    BenchmarkResult { nsg, time_us: median }
}

fn compute_adaptive_nsg(seq_k: usize) -> usize {
    // Adaptive NSG based on seq_k
    if seq_k <= 128 {
        1
    } else if seq_k <= 256 {
        2
    } else if seq_k <= 512 {
        4
    } else if seq_k <= 1024 {
        4
    } else if seq_k <= 2048 {
        8
    } else {
        8
    }
}

fn get_block_size(fmt: &str) -> usize {
    match fmt {
        "f16" | "bf16" => 2,              // 2 bytes per element
        "q4_0" | "q4_1" | "iq4_nl" => 18, // 16 bytes qs + 2 bytes d (and m for q4_1)
        "q5_0" => 22, // 16 bytes qs + 4 bytes qh + 2 bytes d
        "q5_1" => 24, // 16 bytes qs + 4 bytes qh + 2 bytes d + 2 bytes m
        "q8_0" => 34, // 32 bytes qs + 2 bytes d
        "q2_K" => 84,
        "q3_K" => 110,
        "q4_K" => 144,
        "q5_K" => 176,
        "q6_K" => 210,
        _ => 2, // default to f16
    }
}

fn create_test_data(
    fmt: &str,
    seq_k: usize,
    block_size: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // Query: always F16 (seq_q=1)
    let q_size = NUM_HEADS * HEAD_DIM * 2;
    let q_data: Vec<u8> = vec![0x00; q_size]; // F16 zeros

    // K/V: format-specific
    let num_blocks = seq_k * NUM_HEADS * HEAD_DIM / 32;
    let kv_size = num_blocks * block_size;
    let k_data: Vec<u8> = vec![0x00; kv_size];
    let v_data: Vec<u8> = vec![0x00; kv_size];

    (q_data, k_data, v_data)
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
