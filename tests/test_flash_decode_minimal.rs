// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Flash decode correctness and performance test

use half::f16;
use metal::*;
use std::time::Instant;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source =
        include_str!("../src/ops/metal/attention/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).unwrap();

    println!("=== Flash Decode Test ===\n");

    // Config
    let head_dim = 128;
    let num_heads = 32;
    let seq_k = 256;
    let iterations = 500;

    // Query: float format
    let q_data: Vec<f32> =
        (0..num_heads * head_dim).map(|i| (i % 10) as f32 * 0.1).collect();

    // K/V: f16 format
    let k_data: Vec<f16> = (0..seq_k * head_dim)
        .map(|i| f16::from_f32((i % 20) as f32 * 0.05))
        .collect();
    let v_data: Vec<f16> = (0..seq_k * head_dim)
        .map(|i| f16::from_f32((i % 15) as f32 * 0.03))
        .collect();

    // CPU reference for first head only (for correctness check)
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut scores: Vec<f32> = vec![0.0; seq_k];
    for k in 0..seq_k {
        for d in 0..head_dim {
            scores[k] += q_data[d] * k_data[k * head_dim + d].to_f32();
        }
        scores[k] *= scale;
    }

    let mut max_val = f32::NEG_INFINITY;
    let mut sum_val = 0.0f32;
    let mut ref_output = vec![0.0f32; head_dim];

    for k in 0..seq_k {
        let old_max = max_val;
        max_val = max_val.max(scores[k]);

        if old_max > f32::NEG_INFINITY / 2.0 {
            let scale_factor = (old_max - max_val).exp();
            sum_val *= scale_factor;
            for d in 0..head_dim {
                ref_output[d] *= scale_factor;
            }
        }

        let weight = (scores[k] - max_val).exp();
        sum_val += weight;

        for d in 0..head_dim {
            ref_output[d] += weight * v_data[k * head_dim + d].to_f32();
        }
    }

    for d in 0..head_dim {
        ref_output[d] /= sum_val;
    }

    println!("CPU max={}, sum={}", max_val, sum_val);
    println!(
        "CPU output[0:4]: {:.6} {:.6} {:.6} {:.6}",
        ref_output[0], ref_output[1], ref_output[2], ref_output[3]
    );

    // GPU
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
        (num_heads * head_dim * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let args = FlashAttnArgs {
        batch: 1,
        heads: num_heads as u32,
        seq_q: 1,
        seq_k: seq_k as u32,
        head_dim: head_dim as u32,
        scale: scale,
        q_stride_b: (num_heads * head_dim) as u32,
        q_stride_h: head_dim as u32,
        q_stride_s: head_dim as u32,
        k_stride_b: (seq_k * head_dim * 2) as u32,
        k_stride_h: (head_dim * 2) as u32,
        k_stride_s: (head_dim * 2) as u32,
        v_stride_b: (seq_k * head_dim * 2) as u32,
        v_stride_h: (head_dim * 2) as u32,
        v_stride_s: (head_dim * 2) as u32,
        o_stride_b: (num_heads * head_dim) as u32,
        o_stride_h: head_dim as u32,
        o_stride_s: head_dim as u32,
    };

    let args_bytes = unsafe {
        std::slice::from_raw_parts(
            &args as *const _ as *const u8,
            std::mem::size_of::<FlashAttnArgs>(),
        )
    };

    let kernel_name = "flash_decode_grid8x4_f16";
    let pipeline = library.get_function(kernel_name, None).unwrap();
    let pso =
        device.new_compute_pipeline_state_with_function(&pipeline).unwrap();

    // Correctness test
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    enc.set_bytes(0, args_bytes.len() as u64, args_bytes.as_ptr() as *const _);
    enc.set_buffer(1, Some(&q_buf), 0);
    enc.set_buffer(2, Some(&k_buf), 0);
    enc.set_buffer(3, Some(&v_buf), 0);
    enc.set_buffer(4, Some(&out_buf), 0);
    enc.set_threadgroup_memory_length(0, 288);
    enc.dispatch_thread_groups(
        MTLSize::new(1, num_heads as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_output = unsafe {
        std::slice::from_raw_parts(
            out_buf.contents() as *const f32,
            num_heads * head_dim,
        )
    };

    println!(
        "\nGPU output[0:4]: {:.6} {:.6} {:.6} {:.6}",
        gpu_output[0], gpu_output[1], gpu_output[2], gpu_output[3]
    );

    // Check for NaN/inf
    let has_nan = (0..num_heads * head_dim)
        .any(|i| gpu_output[i].is_nan() || gpu_output[i].is_infinite());
    if has_nan {
        println!("WARNING: GPU output contains NaN or inf!");
        for i in 0..head_dim.min(16) {
            if gpu_output[i].is_nan() || gpu_output[i].is_infinite() {
                println!("  gpu_output[{}] = {}", i, gpu_output[i]);
            }
        }
    }

    let max_diff = (0..head_dim)
        .map(|i| (gpu_output[i] - ref_output[i]).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("\nmax_diff={:.6e}", max_diff);

    if max_diff < 1e-4 {
        println!("✓ CORRECT!");
    } else {
        println!("✗ WRONG!");
        return;
    }

    // Performance benchmark
    println!("\n--- Performance Benchmark ---");
    println!(
        "Config: {} heads, {} head_dim, seq_k={}, {} iterations",
        num_heads, head_dim, seq_k, iterations
    );

    // Warmup
    for _ in 0..100 {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_bytes(
            0,
            args_bytes.len() as u64,
            args_bytes.as_ptr() as *const _,
        );
        enc.set_buffer(1, Some(&q_buf), 0);
        enc.set_buffer(2, Some(&k_buf), 0);
        enc.set_buffer(3, Some(&v_buf), 0);
        enc.set_buffer(4, Some(&out_buf), 0);
        enc.set_threadgroup_memory_length(0, 288);
        enc.dispatch_thread_groups(
            MTLSize::new(1, num_heads as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_bytes(
            0,
            args_bytes.len() as u64,
            args_bytes.as_ptr() as *const _,
        );
        enc.set_buffer(1, Some(&q_buf), 0);
        enc.set_buffer(2, Some(&k_buf), 0);
        enc.set_buffer(3, Some(&v_buf), 0);
        enc.set_buffer(4, Some(&out_buf), 0);
        enc.set_threadgroup_memory_length(0, 288);
        enc.dispatch_thread_groups(
            MTLSize::new(1, num_heads as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let elapsed = start.elapsed();

    let total_ops = 4.0
        * seq_k as f64
        * head_dim as f64
        * num_heads as f64
        * iterations as f64;
    let gflops = total_ops / elapsed.as_secs_f64() / 1e9;
    let time_us = elapsed.as_micros() as f64 / iterations as f64;

    println!(
        "\nF16 seq_k={} heads={}: {:.2} GFLOPS ({:.2} µs/iter)",
        seq_k, num_heads, gflops, time_us
    );
}

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
