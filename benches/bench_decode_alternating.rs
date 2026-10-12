// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Alternating benchmark: Original vs Adaptive decode kernel
// Each iteration alternates between kernels to eliminate thermal variance

use metal::*;
use std::time::Instant;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source = include_str!("../src/ops/metal/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).unwrap();

    let num_heads = 32;
    let head_dim = 128;

    // Create test data (F16)
    let q_data: Vec<u16> = vec![0x3C00; num_heads * head_dim]; // 1.0 in FP16

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

    let kernel_orig =
        library.get_function("flash_decode_attention_f16", None).unwrap();
    let pipeline_orig =
        device.new_compute_pipeline_state_with_function(&kernel_orig).unwrap();

    let kernel_adapt = library
        .get_function("flash_decode_attention_f16_adaptive", None)
        .unwrap();
    let pipeline_adapt =
        device.new_compute_pipeline_state_with_function(&kernel_adapt).unwrap();

    println!("=== Alternating Decode Benchmark ===");
    println!("Config: {} heads, {} head_dim\n", num_heads, head_dim);

    // Test different seq_k values
    let seq_k_values = [256, 512, 1024, 2048];

    println!(
        "{:>8} {:>12} {:>12} {:>10} {:>10}",
        "seq_k", "NSG", "Orig (µs)", "Adapt (µs)", "Speedup"
    );
    println!("{}", "-".repeat(60));

    for seq_k in seq_k_values {
        let k_data: Vec<u16> = vec![0x3800; seq_k * num_heads * head_dim];
        let v_data: Vec<u16> = vec![0x3400; seq_k * num_heads * head_dim];

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
        let out_buf_orig = device.new_buffer(
            (num_heads * head_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf_adapt = device.new_buffer(
            (num_heads * head_dim * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let args = Args {
            batch: 1,
            heads: num_heads as u32,
            seq_q: 1,
            seq_k: seq_k as u32,
            head_dim: head_dim as u32,
            scale: 1.0 / (head_dim as f32).sqrt(),
            q_stride_b: (num_heads * head_dim) as u32,
            q_stride_h: head_dim as u32,
            q_stride_s: head_dim as u32,
            k_stride_b: (num_heads * seq_k * head_dim * 2) as u32,
            k_stride_h: (seq_k * head_dim * 2) as u32,
            k_stride_s: (head_dim * 2) as u32,
            v_stride_b: (num_heads * seq_k * head_dim * 2) as u32,
            v_stride_h: (seq_k * head_dim * 2) as u32,
            v_stride_s: (head_dim * 2) as u32,
            o_stride_b: (num_heads * head_dim) as u32,
            o_stride_h: head_dim as u32,
            o_stride_s: head_dim as u32,
        };

        let args_bytes = unsafe {
            std::slice::from_raw_parts(
                &args as *const _ as *const u8,
                std::mem::size_of::<Args>(),
            )
        };

        // Dynamic NSG
        let nsg = if seq_k <= 128 {
            1
        } else if seq_k <= 256 {
            2
        } else if seq_k <= 512 {
            4
        } else if seq_k <= 1024 {
            8
        } else {
            8
        };

        let shmem_size = nsg * (2 + head_dim) * 4;

        // Warmup
        for _ in 0..5 {
            run_decode(
                &queue,
                &pipeline_orig,
                args_bytes,
                &q_buf,
                &k_buf,
                &v_buf,
                &out_buf_orig,
                num_heads,
                1,
                0,
            );
            run_decode(
                &queue,
                &pipeline_adapt,
                args_bytes,
                &q_buf,
                &k_buf,
                &v_buf,
                &out_buf_adapt,
                num_heads,
                nsg,
                shmem_size,
            );
        }

        // Alternating benchmark: 500 rounds
        let num_rounds = 500;
        let mut orig_times = Vec::with_capacity(num_rounds);
        let mut adapt_times = Vec::with_capacity(num_rounds);

        for _ in 0..num_rounds {
            let t0 = Instant::now();
            run_decode(
                &queue,
                &pipeline_orig,
                args_bytes,
                &q_buf,
                &k_buf,
                &v_buf,
                &out_buf_orig,
                num_heads,
                1,
                0,
            );
            orig_times.push(t0.elapsed().as_nanos() as f64);

            let t1 = Instant::now();
            run_decode(
                &queue,
                &pipeline_adapt,
                args_bytes,
                &q_buf,
                &k_buf,
                &v_buf,
                &out_buf_adapt,
                num_heads,
                nsg,
                shmem_size,
            );
            adapt_times.push(t1.elapsed().as_nanos() as f64);
        }

        // Calculate median times
        orig_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        adapt_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let orig_median = orig_times[num_rounds / 2];
        let adapt_median = adapt_times[num_rounds / 2];
        let speedup = orig_median / adapt_median;

        println!(
            "{:>8} {:>12} {:>12.2} {:>12.2} {:>10.2}x",
            seq_k,
            nsg,
            orig_median / 1000.0,
            adapt_median / 1000.0,
            speedup
        );
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
