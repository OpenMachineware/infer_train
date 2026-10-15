// SSD kernel comparison: ours vs llama.cpp algorithm
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use metal::*;
use std::ffi::c_void;
use std::time::Instant;

// Re-create llama.cpp's args struct
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct SsdArgs {
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
    nb02: u64,
    nb03: u64,
    nb11: u64,
    nb13: u64,
    ns12: i32,
    nb20: u64,
    nb22: u64,
    ns21: i32,
    nb31: u64,
    nb41: u64,
    nb43: u64,
    ns42: i32,
    nb51: u64,
    nb53: u64,
    ns52: i32,
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Test params
    let d_state = 64;
    let head_dim = 64;
    let n_tok = 512;
    let n_head = 1;

    // Create test data with llama.cpp layout
    let x_size = n_tok * head_dim;
    let bc_size = n_tok * d_state;
    let state_size = head_dim * d_state;

    let x: Vec<f32> = (0..x_size).map(|i| 0.1 * (i as f32 % 10.0)).collect();
    let dt: Vec<f32> =
        (0..n_tok).map(|i| 0.001 * (i as f32 % 5.0 + 1.0)).collect();
    let a: Vec<f32> = vec![-0.1];
    let b: Vec<f32> = (0..bc_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();
    let c: Vec<f32> = (0..bc_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();

    // Create buffers
    let x_buf = device.new_buffer_with_data(
        x.as_ptr() as *const c_void,
        (x_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let dt_buf = device.new_buffer_with_data(
        dt.as_ptr() as *const c_void,
        (n_tok * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let a_buf = device.new_buffer_with_data(
        a.as_ptr() as *const c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = device.new_buffer_with_data(
        b.as_ptr() as *const c_void,
        (bc_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let c_buf = device.new_buffer_with_data(
        c.as_ptr() as *const c_void,
        (bc_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_buf = device
        .new_buffer((x_size * 4) as u64, MTLResourceOptions::StorageModeShared);
    let s_buf = device.new_buffer(
        (state_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        ((x_size + state_size) * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Compile our kernel
    let source = include_str!("../src/ops/metal/ops/ssd_optimized.metal");
    let options = CompileOptions::new();
    let lib = device
        .new_library_with_source(source, &options)
        .expect("Failed to compile");
    let func = lib
        .get_function("kernel_ssm_scan_ssd_mma", None)
        .expect("Function not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .expect("Pipeline failed");

    // Warmup
    for _ in 0..10 {
        run_kernel(
            &device,
            &queue,
            &pipeline,
            &x_buf,
            &dt_buf,
            &a_buf,
            &b_buf,
            &c_buf,
            &output_buf,
            d_state,
            head_dim,
            n_tok,
            n_head,
        );
    }

    // Measure
    let iterations = 50;
    let mut times = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        run_kernel(
            &device,
            &queue,
            &pipeline,
            &x_buf,
            &dt_buf,
            &a_buf,
            &b_buf,
            &c_buf,
            &output_buf,
            d_state,
            head_dim,
            n_tok,
            n_head,
        );
        times.push(start.elapsed().as_nanos() as f64);
    }

    let mean_ns = times.iter().sum::<f64>() / iterations as f64;
    let mean_us = mean_ns / 1000.0;

    // Calculate GFLOPS correctly for SSD
    // SSD algorithm (per chunk of CS=64 tokens):
    // 1. dtx: CS * HD multiply
    // 2. SAM (C*B^T via MMA): CS^2 * d_state multiply-add (lower triangular)
    // 3. y = SAM @ dtx: CS * HD * CS multiply-add
    // 4. y += exp_acs * (C @ state): d_state * HD multiply-add
    // 5. state update: d_state * HD * CS multiply-add
    let cs = 64.0;
    let hd = head_dim as f64;
    let nc = d_state as f64;
    let nt = n_tok as f64;

    let flops_per_chunk = cs * hd +                    // dtx
        cs * cs * nc / 2.0 +         // SAM (lower triangular)
        cs * hd * cs +               // y = SAM @ dtx
        nc * hd +                    // y += C @ state
        nc * hd * cs; // state update

    let n_chunks = (nt / cs).ceil();
    let total_flops = flops_per_chunk * n_chunks;
    let gflops = total_flops / (mean_ns / 1e9) / 1e9;

    println!("=== SSD MMA Performance ===");
    println!("d_state={}, head_dim={}, n_tok={}", d_state, head_dim, n_tok);
    println!("time: {:.2} us", mean_us);
    println!("Total FLOPs: {:.2e}", total_flops);
    println!("GFLOPS: {:.2}", gflops);
    println!("llama.cpp: ~8000-10000 GFLOPS");
    println!("Gap: {:.1}x", 9000.0 / gflops);
}

fn run_kernel(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    x_buf: &Buffer,
    dt_buf: &Buffer,
    a_buf: &Buffer,
    b_buf: &Buffer,
    c_buf: &Buffer,
    output_buf: &Buffer, // Reuse buffer
    d_state: usize,
    head_dim: usize,
    n_tok: usize,
    n_head: usize,
) {
    let cs = 64;
    let hd = 64;
    let nsg = 4;

    // Threadgroup memory: 27 KB
    let threadgroup_size = (cs * 3 + cs * hd + nsg * 8 * cs + nsg * 2 * 64) * 4;

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();

    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, None, 0); // s0 - not used
    enc.set_buffer(1, Some(x_buf), 0);
    enc.set_buffer(2, Some(dt_buf), 0);
    enc.set_buffer(3, Some(a_buf), 0);
    enc.set_buffer(4, Some(b_buf), 0);
    enc.set_buffer(5, Some(c_buf), 0);
    enc.set_buffer(7, Some(output_buf), 0);

    let ds = d_state as u32;
    let hd_u32 = head_dim as u32;
    let nh = n_head as u32;
    let nt = n_tok as u32;
    let ns = 1u32;

    enc.set_bytes(8, 4, &ds as *const u32 as *const c_void);
    enc.set_bytes(9, 4, &hd_u32 as *const u32 as *const c_void);
    enc.set_bytes(10, 4, &nh as *const u32 as *const c_void);
    enc.set_bytes(11, 4, &nt as *const u32 as *const c_void);
    enc.set_bytes(12, 4, &ns as *const u32 as *const c_void);

    enc.set_threadgroup_memory_length(0, threadgroup_size as u64);

    let grid = MTLSize { width: 1, height: n_head as u64, depth: 1 };
    let tg = MTLSize { width: 128, height: 1, depth: 1 };

    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}
