// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Test multi-SIMD group flash attention for quantized formats

use half::f16;
use infer_train::quant::kv_cache::{
    quantize_row_q4_0, quantize_row_q8_0, BlockQ4_0, BlockQ8_0,
};
use metal::*;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source = include_str!("../../shaders/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).unwrap();

    println!("=== Multi-SIMD Group Quantized Format Test ===\n");

    let head_dim = 128;
    let num_heads = 32;
    let seq_k = 256;

    // Query
    let q_data: Vec<f32> =
        (0..num_heads * head_dim).map(|i| (i % 10) as f32 * 0.1).collect();

    // K/V: raw f32 data
    let k_raw: Vec<f32> =
        (0..seq_k * head_dim).map(|i| (i % 20) as f32 * 0.05).collect();
    let v_raw: Vec<f32> =
        (0..seq_k * head_dim).map(|i| (i % 15) as f32 * 0.03).collect();

    // Test formats
    test_format(
        "q4_0", &device, &queue, &library, &q_data, &k_raw, &v_raw, head_dim,
        num_heads, seq_k,
    );
    test_format(
        "q8_0", &device, &queue, &library, &q_data, &k_raw, &v_raw, head_dim,
        num_heads, seq_k,
    );
}

fn test_format(
    format: &str,
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    q_data: &[f32],
    k_raw: &[f32],
    v_raw: &[f32],
    head_dim: usize,
    num_heads: usize,
    seq_k: usize,
) {
    println!("=== Format: {} ===", format);

    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // CPU reference (using f16 for simplicity)
    let k_f16: Vec<f16> = k_raw.iter().map(|&x| f16::from_f32(x)).collect();
    let v_f16: Vec<f16> = v_raw.iter().map(|&x| f16::from_f32(x)).collect();

    let mut scores: Vec<f32> = vec![0.0; seq_k];
    for k in 0..seq_k {
        for d in 0..head_dim {
            scores[k] += q_data[d] * k_f16[k * head_dim + d].to_f32();
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
            ref_output[d] += weight * v_f16[k * head_dim + d].to_f32();
        }
    }

    for d in 0..head_dim {
        ref_output[d] /= sum_val;
    }

    println!(
        "CPU output[0:4]: {:.6} {:.6} {:.6} {:.6}",
        ref_output[0], ref_output[1], ref_output[2], ref_output[3]
    );

    // Quantize K/V
    let (k_quant, v_quant, k_stride, v_stride) = match format {
        "q4_0" => {
            let k_quant = quantize_k_v_q4_0(k_raw, 1, 1, seq_k, head_dim);
            let v_quant = quantize_k_v_q4_0(v_raw, 1, 1, seq_k, head_dim);
            let stride = (head_dim / 32) * std::mem::size_of::<BlockQ4_0>();
            (k_quant, v_quant, stride, stride)
        }
        "q8_0" => {
            let k_quant = quantize_k_v_q8_0(k_raw, 1, 1, seq_k, head_dim);
            let v_quant = quantize_k_v_q8_0(v_raw, 1, 1, seq_k, head_dim);
            let stride = (head_dim / 32) * std::mem::size_of::<BlockQ8_0>();
            (k_quant, v_quant, stride, stride)
        }
        _ => panic!("Unknown format: {}", format),
    };

    // GPU buffers
    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        k_quant.as_ptr() as *const _,
        k_quant.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        v_quant.as_ptr() as *const _,
        v_quant.len() as u64,
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
        k_stride_b: (seq_k * k_stride) as u32,
        k_stride_h: k_stride as u32,
        k_stride_s: k_stride as u32,
        v_stride_b: (seq_k * v_stride) as u32,
        v_stride_h: v_stride as u32,
        v_stride_s: v_stride as u32,
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
        ("g1", format!("flash_decode_multi_group_{}_g1", format), 1),
        ("g2", format!("flash_decode_multi_group_{}_g2", format), 2),
        ("g4", format!("flash_decode_multi_group_{}_g4", format), 4),
        ("g8", format!("flash_decode_multi_group_{}_g8", format), 8),
    ];

    for (name, kernel_name, num_groups) in &kernels {
        let pipeline = match library.get_function(kernel_name, None) {
            Ok(p) => p,
            Err(e) => {
                println!("{}: Kernel not found: {}", name, e);
                continue;
            }
        };
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

        // Threadgroup memory
        let query_size = 32 * 16;
        let ms_size = num_groups * 16;
        let out_size = num_groups * 4 * 8 * 16;
        let shmem_size = query_size + ms_size + out_size;

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
            "{}: max_diff = {:.6e} {}",
            name,
            max_diff,
            if max_diff < 1e-2 { "✓" } else { "✗" }
        );

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
    println!();
}

fn quantize_k_v_q4_0(
    data: &[f32],
    _batch: usize,
    _heads: usize,
    seq: usize,
    head_dim: usize,
) -> Vec<u8> {
    let blocks_per_row = head_dim / 32;
    let mut result = Vec::with_capacity(
        seq * blocks_per_row * std::mem::size_of::<BlockQ4_0>(),
    );

    for s in 0..seq {
        let row_data = &data[s * head_dim..(s + 1) * head_dim];
        let mut blocks: Vec<BlockQ4_0> =
            vec![unsafe { std::mem::zeroed() }; blocks_per_row];
        quantize_row_q4_0(row_data, &mut blocks);
        let block_bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                blocks.len() * std::mem::size_of::<BlockQ4_0>(),
            )
        };
        result.extend_from_slice(block_bytes);
    }
    result
}

fn quantize_k_v_q8_0(
    data: &[f32],
    _batch: usize,
    _heads: usize,
    seq: usize,
    head_dim: usize,
) -> Vec<u8> {
    let blocks_per_row = head_dim / 32;
    let mut result = Vec::with_capacity(
        seq * blocks_per_row * std::mem::size_of::<BlockQ8_0>(),
    );

    for s in 0..seq {
        let row_data = &data[s * head_dim..(s + 1) * head_dim];
        let mut blocks: Vec<BlockQ8_0> =
            vec![unsafe { std::mem::zeroed() }; blocks_per_row];
        quantize_row_q8_0(row_data, &mut blocks);
        let block_bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                blocks.len() * std::mem::size_of::<BlockQ8_0>(),
            )
        };
        result.extend_from_slice(block_bytes);
    }
    result
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
