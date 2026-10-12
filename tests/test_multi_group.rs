// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test multi-SIMD group flash attention decode

use half::f16;
use metal::*;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source =
        include_str!("../src/ops/metal/attention/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).unwrap();

    println!("=== Multi-SIMD Group Flash Decode Test ===\n");

    let head_dim = 128;
    let num_heads = 32;
    let seq_k = 256;

    // Query
    let q_data: Vec<f32> =
        (0..num_heads * head_dim).map(|i| (i % 10) as f32 * 0.1).collect();

    // K/V: f16
    let k_data: Vec<f16> = (0..seq_k * head_dim)
        .map(|i| f16::from_f32((i % 20) as f32 * 0.05))
        .collect();
    let v_data: Vec<f16> = (0..seq_k * head_dim)
        .map(|i| f16::from_f32((i % 15) as f32 * 0.03))
        .collect();

    // CPU reference
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

    println!(
        "CPU output[0:4]: {:.6} {:.6} {:.6} {:.6}",
        ref_output[0], ref_output[1], ref_output[2], ref_output[3]
    );

    // GPU buffers
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

    // Test different group counts
    let kernels = [
        ("g1", "flash_decode_multi_group_f16_g1", 1),
        ("g2", "flash_decode_multi_group_f16_g2", 2),
        ("g4", "flash_decode_multi_group_f16_g4", 4),
        ("g8", "flash_decode_multi_group_f16_g8", 8),
    ];

    for (name, kernel_name, num_groups) in &kernels {
        let pipeline = library.get_function(kernel_name, None).unwrap();
        let pso =
            device.new_compute_pipeline_state_with_function(&pipeline).unwrap();

        // Clear output
        unsafe {
            std::ptr::write_bytes(
                out_buf.contents() as *mut f32,
                0,
                num_heads * head_dim,
            );
        }

        // Run kernel
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

        // Threadgroup memory:
        // Query: 32 float4s = 128 bytes
        // M and S: NUM_GROUPS * 2 floats = NUM_GROUPS * 8 bytes (need 16-byte alignment)
        // Output: NUM_GROUPS * ITER * 8 float4s = NUM_GROUPS * 128 bytes
        let query_size = 32 * 16; // 32 float4s
        let ms_size = num_groups * 16; // Per-group M and S, 16-byte aligned
        let out_size = num_groups * 4 * 8 * 16; // NUM_GROUPS * ITER * 8 float4s
        let shmem_size = query_size + ms_size + out_size;
        enc.set_threadgroup_memory_length(0, shmem_size as u64);

        enc.dispatch_thread_groups(
            MTLSize::new(1, num_heads as u64, 1),
            MTLSize::new(32, *num_groups as u64, 1),
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

        let max_diff = (0..head_dim)
            .map(|i| (gpu_output[i] - ref_output[i]).abs())
            .fold(0.0f32, |a, b| a.max(b));

        println!(
            "\n{}: output[0:4] = {:.6} {:.6} {:.6} {:.6}",
            name, gpu_output[0], gpu_output[1], gpu_output[2], gpu_output[3]
        );
        println!("{}: max_diff = {:.6e}", name, max_diff);

        if max_diff < 5e-3 {
            // Allow 0.5% error for flash attention numerical stability
            println!("{}: ✓ PASS", name);
        } else {
            println!("{}: ✗ FAIL", name);
        }

        // Performance benchmark
        let iterations = 200;
        for _ in 0..50 {
            // Warmup
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
            enc.set_threadgroup_memory_length(0, shmem_size as u64);
            enc.dispatch_thread_groups(
                MTLSize::new(1, num_heads as u64, 1),
                MTLSize::new(32, *num_groups as u64, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let start = std::time::Instant::now();
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
            enc.set_threadgroup_memory_length(0, shmem_size as u64);
            enc.dispatch_thread_groups(
                MTLSize::new(1, num_heads as u64, 1),
                MTLSize::new(32, *num_groups as u64, 1),
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

        println!("{}: {:.2} GFLOPS ({:.2} µs/iter)", name, gflops, time_us);
    }
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
