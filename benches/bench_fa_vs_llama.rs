// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Compare our Flash Attention vs llama.cpp's implementation

use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 200;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    // Load our self-written kernels
    let our_source = include_str!("../src/ops/metal/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library =
        device.new_library_with_source(our_source, &options).unwrap();

    // Create MetalContext for llama.cpp kernels
    let ctx = MetalContext::new().expect("Failed to create MetalContext");

    println!("=== Flash Attention: Ours vs llama.cpp ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds\n",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );

    // ==================== DECODE (seq_q=1) ====================
    println!("=== DECODE (seq_q = 1) ===\n");

    println!("Format: F16");
    println!(
        "{:>8} {:>6} {:>12} {:>12} {:>10}",
        "seq_k", "NSG", "Ours (µs)", "llama (µs)", "Ratio"
    );
    println!("{}", "-".repeat(55));

    for seq_k in [256, 512, 1024, 2048] {
        let result =
            benchmark_decode_f16(&device, &queue, &our_library, &ctx, seq_k);
        println!(
            "{:>8} {:>6} {:>12.2} {:>12.2} {:>9.1}%",
            seq_k, result.nsg, result.our_us, result.llama_us, result.ratio_pct
        );
    }

    // ==================== GFLOPS comparison ====================
    println!("\n=== GFLOPS (F16 Decode) ===\n");

    println!("{:>8} {:>12} {:>12}", "seq_k", "Ours", "llama.cpp");
    for seq_k in [256, 512, 1024, 2048] {
        let result =
            benchmark_decode_f16(&device, &queue, &our_library, &ctx, seq_k);
        let our_gflops =
            2.0 * NUM_HEADS as f64 * seq_k as f64 * HEAD_DIM as f64
                / result.our_us
                / 1e3;
        let llama_gflops =
            2.0 * NUM_HEADS as f64 * seq_k as f64 * HEAD_DIM as f64
                / result.llama_us
                / 1e3;
        println!("{:>8} {:>12.2} {:>12.2}", seq_k, our_gflops, llama_gflops);
    }
}

struct BenchmarkResult {
    nsg: usize,
    our_us: f64,
    llama_us: f64,
    ratio_pct: f64,
}

fn benchmark_decode_f16(
    device: &Device,
    queue: &CommandQueue,
    our_library: &Library,
    ctx: &MetalContext,
    seq_k: usize,
) -> BenchmarkResult {
    // Create F16 data
    let q_data: Vec<f32> = vec![1.0; NUM_HEADS * HEAD_DIM];
    let k_data: Vec<u16> = vec![0x3800; seq_k * NUM_HEADS * HEAD_DIM]; // 0.5 in FP16
    let v_data: Vec<u16> = vec![0x3400; seq_k * NUM_HEADS * HEAD_DIM]; // 0.25 in FP16

    // NSG based on seq_k
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

    // Our kernel
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

    let our_kernel = our_library
        .get_function("flash_decode_attention_f16_adaptive", None)
        .unwrap();
    let our_pipeline =
        device.new_compute_pipeline_state_with_function(&our_kernel).unwrap();

    let args = build_our_args(1, seq_k);
    let args_bytes = unsafe {
        std::slice::from_raw_parts(
            &args as *const _ as *const u8,
            std::mem::size_of::<OurArgs>(),
        )
    };

    // Warmup
    for _ in 0..5 {
        run_our_decode(
            &queue,
            &our_pipeline,
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

    // Benchmark our kernel
    let mut our_times = Vec::with_capacity(NUM_ROUNDS);
    for _ in 0..NUM_ROUNDS {
        let t0 = Instant::now();
        run_our_decode(
            &queue,
            &our_pipeline,
            args_bytes,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            NUM_HEADS,
            nsg,
            shmem_size,
        );
        our_times.push(t0.elapsed().as_nanos() as f64);
    }
    our_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Benchmark llama.cpp via MetalContext
    let mut llama_times = Vec::with_capacity(NUM_ROUNDS);
    for _ in 0..NUM_ROUNDS {
        let t0 = Instant::now();
        let _ = ctx.flash_attn_f16(
            1, NUM_HEADS, NUM_HEADS, HEAD_DIM, seq_k, &q_data, &k_data, &v_data,
        );
        llama_times.push(t0.elapsed().as_nanos() as f64);
    }
    llama_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let our_us = our_times[NUM_ROUNDS / 2] / 1000.0;
    let llama_us = llama_times[NUM_ROUNDS / 2] / 1000.0;
    let ratio = our_us / llama_us * 100.0;

    BenchmarkResult { nsg, our_us, llama_us, ratio_pct: ratio }
}

#[repr(C)]
struct OurArgs {
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

fn build_our_args(seq_q: usize, seq_k: usize) -> OurArgs {
    OurArgs {
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

fn run_our_decode(
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
