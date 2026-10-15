// SSM operator comparison with llama.cpp kernels
// Direct benchmark against llama.cpp's Metal kernels
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use metal::*;
use std::ffi::c_void;
use std::time::Instant;

// Common header needed by llama.cpp kernels
const LLAMA_COMMON_HEADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32

#define FOR_UNROLL(expr) for (int i = 0; i < (expr); ++i)
"#;

// Recreate ggml_metal_kargs_ssm_scan from llama.cpp
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct GgmlMetalKargsSsmScan {
    d_state: i32,
    d_inner: i32,
    n_head: i32,
    n_group: i32,
    n_seq_tokens: i32,
    n_seq_tokens_total: i32,
    token_offset: i32,
    n_seqs: i32,
    K: i32,
    s_off: u64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    ns12: i32,
    nb13: u64,
    nb20: u64,
    nb21: u64,
    ns21: i32,
    nb22: u64,
    nb23: u64,
    ne30: i32,
    nb31: u64,
    nb40: u64,
    nb41: u64,
    nb42: u64,
    ns42: i32,
    nb43: u64,
    nb50: u64,
    nb51: u64,
    nb52: u64,
    ns52: i32,
    nb53: u64,
}

fn compute_scan_gflops(
    d_inner: usize,
    n_tok: usize,
    d_state: usize,
    time_ns: f64,
) -> f64 {
    // Each token: 2*d_state FMA for state update + d_state FMA for output = 3*d_state FMA
    // Plus: softplus (exp, log), exp(dt*A) = 3 exp/log per token
    // Approx: 3*d_state + 3 = 3*(d_state+1) ops per token per d_inner
    let ops_per_token = 3.0 * (d_state + 1) as f64;
    let total_ops = ops_per_token * n_tok as f64 * d_inner as f64;
    total_ops / (time_ns / 1e9) / 1e9
}

fn main() {
    let device = Device::system_default().expect("No Metal device found");
    let queue = device.new_command_queue();

    // Load both Metal libraries
    let our_lib = device
        .new_library_with_file("src/ops/metal/ops.metallib")
        .expect("Failed to load our Metal library");
    let llama_lib = device.new_library_with_file("llama.cpp-0.6.0/build-metal/ggml-metal.metal/src/ggml-metal.metaldefault.metallib")
        .expect("Failed to load llama.cpp Metal library");

    // Create pipelines
    let our_kernel = our_lib
        .get_function("ssm_scan_d16_f32", None)
        .expect("Failed to get our kernel");
    let our_pipeline = device
        .new_compute_pipeline_state_with_function(&our_kernel)
        .expect("Failed to create our pipeline");

    let llama_kernel = llama_lib
        .get_function("kernel_ssm_scan_f32", None)
        .expect("Failed to get llama.cpp kernel");
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_kernel)
        .expect("Failed to create llama.cpp pipeline");

    // Test parameters
    let d_state = 16usize;
    let d_inner = 4096usize;
    let n_tok = 512usize;
    let n_head = 1usize;
    let n_group = 1usize;
    let n_seqs = 1usize;

    // Create input data
    let x_size = d_inner * n_tok;
    let dt_size = n_head * n_tok;
    let a_size = d_state; // Scalar per head
    let bc_size = d_state * n_tok;
    let state_size = d_inner * d_state;

    let x: Vec<f32> = (0..x_size).map(|i| 0.1 * (i as f32 % 10.0)).collect();
    let dt: Vec<f32> =
        (0..dt_size).map(|i| 0.001 * (i as f32 % 5.0 + 1.0)).collect();
    let a: Vec<f32> =
        (0..a_size).map(|i| -0.1 * ((i as f32 % 10.0) / 10.0)).collect();
    let b: Vec<f32> = (0..bc_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();
    let c: Vec<f32> = (0..bc_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();

    // Create Metal buffers for our kernel
    let x_buffer = device.new_buffer_with_data(
        x.as_ptr() as *const c_void,
        (x_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let dt_buffer = device.new_buffer_with_data(
        dt.as_ptr() as *const c_void,
        (dt_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let a_buffer = device.new_buffer_with_data(
        a.as_ptr() as *const c_void,
        (a_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buffer = device.new_buffer_with_data(
        b.as_ptr() as *const c_void,
        (bc_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let c_buffer = device.new_buffer_with_data(
        c.as_ptr() as *const c_void,
        (bc_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_buffer = device
        .new_buffer((x_size * 4) as u64, MTLResourceOptions::StorageModeShared);
    let s_out_buffer = device.new_buffer(
        (state_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Benchmark our kernel
    println!(
        "\n=== SSM Scan (d_state=16, d_inner={}, n_tok={}) ===",
        d_inner, n_tok
    );

    // Warmup
    for _ in 0..10 {
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&our_pipeline);
        encoder.set_buffer(0, Some(&x_buffer), 0);
        encoder.set_buffer(1, Some(&dt_buffer), 0);
        encoder.set_buffer(2, Some(&a_buffer), 0);
        encoder.set_buffer(3, Some(&b_buffer), 0);
        encoder.set_buffer(4, Some(&c_buffer), 0);
        encoder.set_buffer(5, None, 0); // s0
        encoder.set_buffer(6, Some(&y_buffer), 0);
        encoder.set_buffer(7, Some(&s_out_buffer), 0);
        let d_inner_u32 = d_inner as u32;
        let n_tok_u32 = n_tok as u32;
        encoder.set_bytes(8, 4, &d_inner_u32 as *const u32 as *const c_void);
        encoder.set_bytes(9, 4, &n_tok_u32 as *const u32 as *const c_void);

        let grid_size =
            MTLSize { width: d_inner as u64, height: d_state as u64, depth: 1 };
        let threadgroup_size = MTLSize { width: 32, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Measure
    let iterations = 50;
    let mut our_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&our_pipeline);
        encoder.set_buffer(0, Some(&x_buffer), 0);
        encoder.set_buffer(1, Some(&dt_buffer), 0);
        encoder.set_buffer(2, Some(&a_buffer), 0);
        encoder.set_buffer(3, Some(&b_buffer), 0);
        encoder.set_buffer(4, Some(&c_buffer), 0);
        encoder.set_buffer(5, None, 0);
        encoder.set_buffer(6, Some(&y_buffer), 0);
        encoder.set_buffer(7, Some(&s_out_buffer), 0);
        let d_inner_u32 = d_inner as u32;
        let n_tok_u32 = n_tok as u32;
        encoder.set_bytes(8, 4, &d_inner_u32 as *const u32 as *const c_void);
        encoder.set_bytes(9, 4, &n_tok_u32 as *const u32 as *const c_void);

        let grid_size =
            MTLSize { width: d_inner as u64, height: d_state as u64, depth: 1 };
        let threadgroup_size = MTLSize { width: 32, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        our_times.push(start.elapsed().as_nanos() as f64);
    }

    let our_mean_ns = our_times.iter().sum::<f64>() / iterations as f64;
    let our_mean_us = our_mean_ns / 1000.0;
    let our_gflops = compute_scan_gflops(d_inner, n_tok, d_state, our_mean_ns);

    println!("Our kernel:");
    println!("  time: {:.2} us", our_mean_us);
    println!("  GFLOPS: {:.2}", our_gflops);

    // Now benchmark llama.cpp kernel
    // Need to set up data in llama.cpp's expected layout
    // x: [dim, nh, nt, ns] = [4096, 1, 512, 1]
    // dt: [nh, nt, ns] = [1, 512, 1]
    // A: [d_state, nh] = [16, 1]
    // B: [d_state, ng, nt, ns] = [16, 1, 512, 1]
    // C: [d_state, ng, nt, ns] = [16, 1, 512, 1]

    // Create contiguous buffers in llama.cpp layout
    // For simplicity, use the same buffers - they're already in the right layout for this test

    // Set up strides for llama.cpp
    let args = GgmlMetalKargsSsmScan {
        d_state: d_state as i32,
        d_inner: d_inner as i32,
        n_head: n_head as i32,
        n_group: n_group as i32,
        n_seq_tokens: n_tok as i32,
        n_seq_tokens_total: n_tok as i32,
        token_offset: 0,
        n_seqs: n_seqs as i32,
        K: 0,                       // No state caching
        s_off: (x_size * 4) as u64, // y output before state
        // s0 (src0): [d_inner, d_state] = [4096, 16]
        nb00: 4,
        nb01: (d_state * 4) as u64,
        nb02: (d_inner * d_state * 4) as u64,
        nb03: (d_inner * d_state * 4) as u64,
        // x (src1): [d_inner, n_head, n_tok, n_seqs] = [4096, 1, 512, 1]
        nb10: 4,
        nb11: (d_inner * 4) as u64,
        nb12: (d_inner * n_head * 4) as u64,
        nb13: (d_inner * n_head * n_tok * 4) as u64,
        ns12: (n_head * d_inner) as i32, // nb12/nb10
        // dt (src2): [n_head, n_tok, n_seqs] = [1, 512, 1]
        nb20: 4,
        nb21: (n_head * 4) as u64,
        nb22: (n_head * n_tok * 4) as u64,
        nb23: (n_head * n_tok * n_seqs * 4) as u64,
        ns21: n_head as i32, // nb21/nb20
        // A (src3): [d_state, n_head] = [16, 1]
        ne30: d_state as i32,
        nb31: (d_state * 4) as u64,
        // B (src4): [d_state, n_group, n_tok, n_seqs] = [16, 1, 512, 1]
        nb40: 4,
        nb41: (d_state * 4) as u64,
        nb42: (d_state * n_group * 4) as u64,
        nb43: (d_state * n_group * n_tok * 4) as u64,
        ns42: (n_group * d_state) as i32, // nb42/nb40
        // C (src5): [d_state, n_group, n_tok, n_seqs] = [16, 1, 512, 1]
        nb50: 4,
        nb51: (d_state * 4) as u64,
        nb52: (d_state * n_group * 4) as u64,
        nb53: (d_state * n_group * n_tok * 4) as u64,
        ns52: (n_group * d_state) as i32, // nb52/nb50
    };

    // Create combined output buffer for llama.cpp (y + s_buff)
    let llama_output_size = x_size + state_size;
    let llama_output = device.new_buffer(
        (llama_output_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create ids buffer (for seq indexing, use 0)
    let ids: Vec<i32> = vec![0];
    let ids_buffer = device.new_buffer_with_data(
        ids.as_ptr() as *const c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    // Warmup llama.cpp
    for _ in 0..10 {
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&llama_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<GgmlMetalKargsSsmScan>() as u64,
            &args as *const GgmlMetalKargsSsmScan as *const c_void,
        );
        encoder.set_buffer(1, Some(&s_out_buffer), 0); // src0 (s0)
        encoder.set_buffer(2, Some(&x_buffer), 0); // src1 (x)
        encoder.set_buffer(3, Some(&dt_buffer), 0); // src2 (dt)
        encoder.set_buffer(4, Some(&a_buffer), 0); // src3 (A)
        encoder.set_buffer(5, Some(&b_buffer), 0); // src4 (B)
        encoder.set_buffer(6, Some(&c_buffer), 0); // src5 (C)
        encoder.set_buffer(7, Some(&ids_buffer), 0); // src6 (ids)
        encoder.set_buffer(8, Some(&llama_output), 0); // dst

        // Threadgroup memory for shared x_dt and dA
        let threadgroup_size = 128 * 4 * 4; // sgptg*32 + sgptg + sgptg floats
        encoder.set_threadgroup_memory_length(0, threadgroup_size as u64);

        // Grid: [d_inner, n_head, n_seqs]
        let grid_size = MTLSize {
            width: d_inner as u64,
            height: n_head as u64,
            depth: n_seqs as u64,
        };
        // Threadgroup: 128 threads (4 SIMD groups of 32)
        let threadgroup_size = MTLSize { width: 128, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Measure llama.cpp
    let mut llama_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&llama_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<GgmlMetalKargsSsmScan>() as u64,
            &args as *const GgmlMetalKargsSsmScan as *const c_void,
        );
        encoder.set_buffer(1, Some(&s_out_buffer), 0);
        encoder.set_buffer(2, Some(&x_buffer), 0);
        encoder.set_buffer(3, Some(&dt_buffer), 0);
        encoder.set_buffer(4, Some(&a_buffer), 0);
        encoder.set_buffer(5, Some(&b_buffer), 0);
        encoder.set_buffer(6, Some(&c_buffer), 0);
        encoder.set_buffer(7, Some(&ids_buffer), 0);
        encoder.set_buffer(8, Some(&llama_output), 0);

        let threadgroup_size = 128 * 4 * 4;
        encoder.set_threadgroup_memory_length(0, threadgroup_size as u64);

        let grid_size = MTLSize {
            width: d_inner as u64,
            height: n_head as u64,
            depth: n_seqs as u64,
        };
        let threadgroup_size = MTLSize { width: 128, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        llama_times.push(start.elapsed().as_nanos() as f64);
    }

    let llama_mean_ns = llama_times.iter().sum::<f64>() / iterations as f64;
    let llama_mean_us = llama_mean_ns / 1000.0;
    let llama_gflops =
        compute_scan_gflops(d_inner, n_tok, d_state, llama_mean_ns);

    println!("\nllama.cpp kernel:");
    println!("  time: {:.2} us", llama_mean_us);
    println!("  GFLOPS: {:.2}", llama_gflops);

    println!("\nComparison:");
    println!("  Ratio (ours/llama.cpp): {:.2}x", our_mean_us / llama_mean_us);
    println!("  GFLOPS ratio: {:.2}x", our_gflops / llama_gflops);
}
