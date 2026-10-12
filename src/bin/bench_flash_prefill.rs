// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Benchmark Flash Prefill GPU performance

use metal::*;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    // Load our kernels
    let source = include_str!("../../shaders/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);

    println!("Compiling shader ({} lines)...", source.lines().count());
    let start = std::time::Instant::now();
    let library = match device.new_library_with_source(source, &options) {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("Failed to compile: {}", e);
            std::process::exit(1);
        }
    };
    println!("Compilation took: {:.2}s", start.elapsed().as_secs_f64());

    // Prefill kernel test with different sizes
    let func =
        library.get_function("flash_prefill_attention_f16", None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&func).unwrap();

    // Test sizes
    let test_cases = [(16, 256), (64, 512), (128, 1024), (256, 2048)];

    println!("\n=== Flash Prefill F16 Benchmark ===");
    println!("seq_q  seq_k   Time(µs)  GFLOPS");

    for (seq_q, seq_k) in test_cases {
        let (time_us, gflops) =
            run_prefill_test(&device, &queue, &pipeline, seq_q, seq_k);
        println!("{:4}  {:5}  {:8.2}  {:7.2}", seq_q, seq_k, time_us, gflops);
    }
}

fn run_prefill_test(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    seq_q: usize,
    seq_k: usize,
) -> (f64, f64) {
    // Create test data
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

    // FlashAttnArgs struct matching kernel definition
    #[repr(C)]
    struct FlashAttnArgs {
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

    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let args = FlashAttnArgs {
        batch: 1,
        heads: NUM_HEADS as u32,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        head_dim: HEAD_DIM as u32,
        scale,
        q_stride_b: (NUM_HEADS * HEAD_DIM) as u32,
        q_stride_h: HEAD_DIM as u32,
        q_stride_s: HEAD_DIM as u32,
        k_stride_b: (NUM_HEADS * HEAD_DIM) as u32,
        k_stride_h: HEAD_DIM as u32,
        k_stride_s: HEAD_DIM as u32,
        v_stride_b: (NUM_HEADS * HEAD_DIM) as u32,
        v_stride_h: HEAD_DIM as u32,
        v_stride_s: HEAD_DIM as u32,
        o_stride_b: (NUM_HEADS * HEAD_DIM) as u32,
        o_stride_h: HEAD_DIM as u32,
        o_stride_s: HEAD_DIM as u32,
    };

    // Dispatch configuration
    let threads_per_group = MTLSize::new(128, 1, 1);
    let num_groups = MTLSize::new(seq_q as u64, NUM_HEADS as u64, 1);

    // Warmup
    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_bytes(
        0,
        std::mem::size_of::<FlashAttnArgs>() as u64,
        &args as *const FlashAttnArgs as *const std::ffi::c_void,
    );
    encoder.set_buffer(1, Some(&q_buf), 0);
    encoder.set_buffer(2, Some(&k_buf), 0);
    encoder.set_buffer(3, Some(&v_buf), 0);
    encoder.set_buffer(4, Some(&out_buf), 0);
    encoder.dispatch_thread_groups(num_groups, threads_per_group);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Benchmark
    let iterations = 10;
    let mut total_time = 0.0f64;

    for _ in 0..iterations {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<FlashAttnArgs>() as u64,
            &args as *const FlashAttnArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&q_buf), 0);
        encoder.set_buffer(2, Some(&k_buf), 0);
        encoder.set_buffer(3, Some(&v_buf), 0);
        encoder.set_buffer(4, Some(&out_buf), 0);
        encoder.dispatch_thread_groups(num_groups, threads_per_group);
        encoder.end_encoding();

        let start = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        total_time += start.elapsed().as_nanos() as f64 / 1000.0;
    }

    let time_us = total_time / iterations as f64;
    let flops =
        2.0 * seq_q as f64 * seq_k as f64 * HEAD_DIM as f64 * NUM_HEADS as f64;
    let gflops = flops / time_us / 1000.0;

    (time_us, gflops)
}
