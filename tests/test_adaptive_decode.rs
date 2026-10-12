// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use metal::*;
use std::time::Instant;

fn run_decode_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    args_bytes: &[u8],
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    out_buf: &Buffer,
    num_heads: usize,
    nsg: usize, // Number of SIMD groups
    shmem_size: usize,
) -> Vec<f32> {
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
    // Threadgroup: (32 threads per SIMD, nsg SIMD groups, 1)
    enc.dispatch_thread_groups(
        MTLSize::new(1, num_heads as u64, 1),
        MTLSize::new(32, nsg as u64, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let output: &[f32] = unsafe {
        std::slice::from_raw_parts(
            out_buf.contents() as *const f32,
            num_heads * 128,
        )
    };
    output.to_vec()
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source = include_str!("../src/ops/metal/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).unwrap();

    let num_heads = 32;
    let head_dim = 128;

    // Test with realistic random-ish data for correctness verification
    let mut q_data = vec![0u16; num_heads * head_dim];
    let mut k_data = vec![0u16; 0];
    let mut v_data = vec![0u16; 0];

    // Simple pattern: varying values to make the output non-uniform
    for i in 0..q_data.len() {
        q_data[i] = 0x3C00 + ((i % 16) as u16); // 1.0 + small variations
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

    // Test different seq_k values for dynamic scaling
    let seq_k_values = [64, 128, 256, 512, 1024, 2048];

    // Load both kernels
    let kernel_orig =
        library.get_function("flash_decode_attention_f16", None).unwrap();
    let pipeline_orig =
        device.new_compute_pipeline_state_with_function(&kernel_orig).unwrap();

    let kernel_adapt = library
        .get_function("flash_decode_attention_f16_adaptive", None)
        .unwrap();
    let pipeline_adapt =
        device.new_compute_pipeline_state_with_function(&kernel_adapt).unwrap();

    println!("=== Decode Kernel Dynamic Scaling Test ===");
    println!("Config: {} heads, {} head_dim", num_heads, head_dim);
    println!();
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>10}",
        "seq_k", "NSG", "Time (ms)", "GFLOPS", "Speedup"
    );
    println!("{}", "-".repeat(60));

    for seq_k in seq_k_values {
        // Resize data
        k_data = vec![0x3800; seq_k * num_heads * head_dim]; // 0.5 in FP16
        v_data = vec![0x3400; seq_k * num_heads * head_dim]; // 0.25 in FP16

        // Add variation to make outputs non-uniform
        for i in 0..k_data.len() {
            k_data[i] = 0x3800 + ((i % 32) as u16);
            v_data[i] = 0x3400 + ((i % 64) as u16);
        }

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

        // Dynamic NSG: scale with seq_k
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
        }; // Cap at 8 SIMD groups (256 threads)

        let shmem_size = nsg * (2 + head_dim) * 4; // nsg * (max + sum + head_dim output)

        // Benchmark original (always 1 SIMD group = 32 threads)
        // Warmup
        for _ in 0..5 {
            run_decode_kernel(
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
        }
        let start = Instant::now();
        for _ in 0..100 {
            run_decode_kernel(
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
        }
        let orig_time = start.elapsed();

        // Benchmark adaptive with dynamic NSG
        // Warmup
        for _ in 0..5 {
            run_decode_kernel(
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
        let start = Instant::now();
        for _ in 0..100 {
            run_decode_kernel(
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
        let adapt_time = start.elapsed();

        let gflops =
            2.0 * num_heads as f64 * seq_k as f64 * head_dim as f64 * 100.0;
        let orig_gflops = gflops / orig_time.as_secs_f64() / 1e9;
        let adapt_gflops = gflops / adapt_time.as_secs_f64() / 1e9;
        let speedup = orig_time.as_secs_f64() / adapt_time.as_secs_f64();

        println!(
            "{:>8} {:>12} {:>12.2} {:>12.2} {:>10.2}x",
            seq_k,
            nsg,
            adapt_time.as_millis(),
            adapt_gflops,
            speedup
        );

        // Verify correctness: compare outputs
        let orig_output = unsafe {
            std::slice::from_raw_parts(
                out_buf_orig.contents() as *const f32,
                num_heads * head_dim,
            )
        };
        let adapt_output = unsafe {
            std::slice::from_raw_parts(
                out_buf_adapt.contents() as *const f32,
                num_heads * head_dim,
            )
        };

        let mut max_diff = 0.0f32;
        let mut max_diff_idx = 0;
        for i in 0..num_heads * head_dim {
            let diff = (orig_output[i] - adapt_output[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_diff_idx = i;
            }
        }

        if max_diff > 1e-4 {
            println!(
                "  ⚠️  Output mismatch: max_diff = {:.6} at idx {}",
                max_diff, max_diff_idx
            );
            println!(
                "  orig[{}] = {:.6}, adapt[{}] = {:.6}",
                max_diff_idx,
                orig_output[max_diff_idx],
                max_diff_idx,
                adapt_output[max_diff_idx]
            );
        }
    }
}
