// SSM Scan direct comparison: ours vs llama.cpp
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use metal::*;
use std::ffi::c_void;
use std::time::Instant;

const N_SIMDWIDTH: usize = 32;

// llama.cpp args struct
#[repr(C)]
struct LlamaArgs {
    d_state: i64,
    d_inner: i64,
    n_head: i64,
    n_group: i64,
    n_seq_tokens: i64,
    n_seq_tokens_total: i64,
    token_offset: i64,
    n_seqs: i64,
    K: i64,
    s_off: u64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    ns12: u64,
    nb13: u64,
    nb20: u64,
    nb21: u64,
    ns21: u64,
    nb22: u64,
    ne30: i64,
    nb31: u64,
    nb41: u64,
    nb42: u64,
    ns42: u64,
    nb43: u64,
    nb51: u64,
    nb52: u64,
    ns52: u64,
    nb53: u64,
    nb0: u64,
}

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Compile our kernel
    let our_source = include_str!("../src/ops/metal/ops/ssm_scan.metal");
    let options = CompileOptions::new();
    let our_lib = device
        .new_library_with_source(our_source, &options)
        .expect("Failed to compile our kernel");
    let our_func = our_lib
        .get_function("ssm_scan_d16_f32", None)
        .expect("Our function not found");
    let our_pipeline = device
        .new_compute_pipeline_state_with_function(&our_func)
        .expect("Our pipeline failed");

    // Compile llama.cpp kernel
    let llama_common = r#"
#include <metal_stdlib>
using namespace metal;
#define N_SIMDWIDTH 32

typedef struct {
    int64_t  d_state;
    int64_t  d_inner;
    int64_t  n_head;
    int64_t  n_group;
    int64_t  n_seq_tokens;
    int64_t  n_seq_tokens_total;
    int64_t  token_offset;
    int64_t  n_seqs;
    int64_t  K;
    uint64_t s_off;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t ns12;
    uint64_t nb13;
    uint64_t nb20;
    uint64_t nb21;
    uint64_t ns21;
    uint64_t nb22;
    int64_t  ne30;
    uint64_t nb31;
    uint64_t nb41;
    uint64_t nb42;
    uint64_t ns42;
    uint64_t nb43;
    uint64_t nb51;
    uint64_t nb52;
    uint64_t ns52;
    uint64_t nb53;
    uint64_t nb0;
} ggml_metal_kargs_ssm_scan;
"#;
    let llama_kernel = r#"
template<bool TAIL>
kernel void kernel_ssm_scan_impl(
        constant ggml_metal_kargs_ssm_scan & args,
        device const void * src0,
        device const void * src1,
        device const void * src2,
        device const void * src3,
        device const void * src4,
        device const void * src5,
        device const void * src6,
        device      float * dst,
        threadgroup float * shared [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgptg[[simdgroups_per_threadgroup]],
        uint3    tgpg[[threadgroups_per_grid]]) {
    constexpr short NW = N_SIMDWIDTH;
    threadgroup float * shared_sums = shared;
    threadgroup float * shared_x_dt = shared + sgptg * NW;
    threadgroup float * shared_dA   = shared + sgptg * NW + sgptg;

    shared_sums[tpitg.x] = 0.0f;

    const int32_t i0 = tpitg.x;
    const int32_t i1 = tgpig.x;
    const int32_t ir = tgpig.y;
    const int32_t i3 = tgpig.z;

    const int32_t nc  = args.d_state;
    const int32_t nr  = args.d_inner;
    const int32_t nh  = args.n_head;
    const int32_t ng  = args.n_group;
    const int32_t n_t = args.n_seq_tokens;
    const int32_t n_s = args.n_seqs;
    const int32_t K   = args.K;
    const int32_t n_t_total = TAIL ? args.n_seq_tokens_total : n_t;
    const int32_t t_off     = TAIL ? args.token_offset       : 0;

    const int32_t s_off = args.s_off;

    device const int32_t * ids = (device const int32_t *) src6;

    device       float * s_buff  = (device       float *) ((device       char *) dst  + ir*args.nb02 +      i3*args.nb03 + s_off);
    device const float * s0_buff = t_off != 0 ?
        s_buff :
        (device const float *) ((device const char *) src0 + ir*args.nb02 + ids[i3]*args.nb03);

    const int32_t i = i0 + i1*nc;
    const int32_t g = ir / (nh / ng);

    float s0 = s0_buff[i];
    float s  = 0.0f;

    device const float * A = (device const float *) ((device const char *) src3 + ir*args.nb31);

    const float A0 = A[i0%args.ne30];

    device const float * x  = (device const float *)((device const char *) src1 + i1*args.nb10 + ir*args.nb11 + t_off*args.nb12 + i3*args.nb13);
    device const float * dt = (device const float *)((device const char *) src2 + ir*args.nb20 + t_off*args.nb21 + i3*args.nb22);
    device const float * B  = (device const float *)((device const char *) src4 + g*args.nb41 + t_off*args.nb42 + i3*args.nb43);
    device const float * C  = (device const float *)((device const char *) src5 + g*args.nb51 + t_off*args.nb52 + i3*args.nb53);

    device float * y = dst + (i1 + ir*nr + t_off*nh*nr + i3*(n_t_total*nh*nr));

    for (int i2 = 0; i2 < n_t; i2 += sgptg) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (i0 < sgptg && i2 + i0 < n_t) {
            device const float * x_t  = x  + i0 * args.ns12;
            device const float * dt_t = dt + i0 * args.ns21;

            const float dt0  = dt_t[0];
            const float dtsp = dt0 <= 20.0f ? log(1.0f + exp(dt0)) : dt0;
            shared_x_dt[i0] = x_t[0] * dtsp;
            shared_dA[i0]   = dtsp;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < sgptg && i2 + t < n_t; t++) {
            const float x_dt = shared_x_dt[t];
            const float dA   = exp(shared_dA[t] * A0);

            s = (s0 * dA) + (B[i0] * x_dt);

            const float sumf = simd_sum(s * C[i0]);

            if (tiisg == 0) {
                shared_sums[t*NW + sgitg] = sumf;
            }

            s0 = s;

            B  += args.ns42;
            C  += args.ns52;
        }

        x  += sgptg * args.ns12;
        dt += sgptg * args.ns21;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float sumf = simd_sum(shared_sums[sgitg*NW + tiisg]);

        if (tiisg == 0 && i2 + sgitg < n_t) {
            y[sgitg*nh*nr] = sumf;
        }

        y += sgptg*nh*nr;
    }

    s_buff[i] = s;
}

template [[host_name("kernel_ssm_scan_f32")]] kernel void kernel_ssm_scan_impl<false>(constant ggml_metal_kargs_ssm_scan &, device const void *, device const void *, device const void *, device const void *, device const void *, device const void *, device const void *, device float *, threadgroup float *, uint3, ushort3, ushort, ushort, ushort, uint3);
"#;
    let llama_full = format!("{}\n{}", llama_common, llama_kernel);

    println!("=== SSM Scan Performance Comparison ===\n");

    // Test parameters for Mamba-1 (d_state=16)
    let d_state: usize = 16;
    let d_inner: usize = 4096; // Mamba-1 typical
    let n_tok: usize = 512;
    let n_head: usize = 1;

    // Create test data with our layout
    // Our layout: x [d_inner, n_tok], A [d_inner, d_state], B/C [d_state, n_tok]
    let x_size = d_inner * n_tok;
    let a_size = d_inner * d_state;
    let bc_size = d_state * n_tok;
    let y_size = d_inner * n_tok;
    let state_size = d_inner * d_state;

    let x: Vec<f32> = (0..x_size).map(|i| 0.1 * (i as f32 % 10.0)).collect();
    let dt: Vec<f32> = (0..d_inner)
        .flat_map(|_| (0..n_tok).map(|j| 0.001 * (j as f32 % 5.0 + 1.0)))
        .collect();
    let a: Vec<f32> =
        (0..a_size).map(|i| -0.1 * ((i % 16) as f32 + 1.0)).collect();
    let b: Vec<f32> = (0..bc_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();
    let c: Vec<f32> = (0..bc_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();

    // Create buffers for our kernel
    let x_buf = device.new_buffer_with_data(
        x.as_ptr() as *const c_void,
        (x_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let dt_buf = device.new_buffer_with_data(
        dt.as_ptr() as *const c_void,
        (d_inner * n_tok * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let a_buf = device.new_buffer_with_data(
        a.as_ptr() as *const c_void,
        (a_size * 4) as u64,
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
        .new_buffer((y_size * 4) as u64, MTLResourceOptions::StorageModeShared);
    let s_buf = device.new_buffer(
        (state_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Warmup our kernel
    for _ in 0..10 {
        run_our_kernel(
            &queue,
            &our_pipeline,
            &x_buf,
            &dt_buf,
            &a_buf,
            &b_buf,
            &c_buf,
            &y_buf,
            &s_buf,
            d_inner,
            n_tok,
        );
    }

    // Measure our kernel
    let iterations = 50;
    let mut our_times = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        run_our_kernel(
            &queue,
            &our_pipeline,
            &x_buf,
            &dt_buf,
            &a_buf,
            &b_buf,
            &c_buf,
            &y_buf,
            &s_buf,
            d_inner,
            n_tok,
        );
        our_times.push(start.elapsed().as_nanos() as f64);
    }

    let our_mean_ns = our_times.iter().sum::<f64>() / iterations as f64;
    let our_mean_us = our_mean_ns / 1000.0;

    println!("Our kernel (ssm_scan_d16_f32):");
    println!("  Time: {:.2} us", our_mean_us);

    // Try to compile llama.cpp kernel
    match device.new_library_with_source(&llama_full, &options) {
        Ok(llama_lib) => {
            match llama_lib.get_function("kernel_ssm_scan_f32", None) {
                Ok(llama_func) => {
                    match device
                        .new_compute_pipeline_state_with_function(&llama_func)
                    {
                        Ok(llama_pipeline) => {
                            println!(
                                "\nllama.cpp kernel available, benchmarking..."
                            );

                            // Create llama.cpp layout buffers
                            // llama.cpp: x [dim, nh, nt, ns], dt [nh, nt, ns], A [ne30, nh]
                            // B/C [d_state, ng, nt, ns]
                            let ll_x_size = d_inner * n_tok;
                            let ll_dt_size = n_tok;
                            let ll_a_size = d_state;
                            let ll_bc_size = d_state * n_tok;

                            let ll_x_buf = device.new_buffer_with_data(
                                x.as_ptr() as *const c_void,
                                (ll_x_size * 4) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );
                            let ll_dt_buf = device.new_buffer_with_data(
                                dt.as_ptr() as *const c_void,
                                (ll_dt_size * 4) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );
                            let ll_a_buf = device.new_buffer_with_data(
                                a.as_ptr() as *const c_void,
                                (ll_a_size * 4) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );
                            let ll_b_buf = device.new_buffer_with_data(
                                b.as_ptr() as *const c_void,
                                (ll_bc_size * 4) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );
                            let ll_c_buf = device.new_buffer_with_data(
                                c.as_ptr() as *const c_void,
                                (ll_bc_size * 4) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );
                            let ll_output_buf = device.new_buffer(
                                ((y_size + state_size) * 4) as u64,
                                MTLResourceOptions::StorageModeShared,
                            );

                            let sgptg = 4; // simdgroups per threadgroup
                            let threadgroup_mem =
                                (sgptg * N_SIMDWIDTH + sgptg * 2) * 4; // shared_sums + shared_x_dt + shared_dA

                            let args = LlamaArgs {
                                d_state: d_state as i64,
                                d_inner: d_inner as i64,
                                n_head: n_head as i64,
                                n_group: 1,
                                n_seq_tokens: n_tok as i64,
                                n_seq_tokens_total: n_tok as i64,
                                token_offset: 0,
                                n_seqs: 1,
                                K: 0,
                                s_off: (y_size) as u64 * 4,
                                nb00: 0,
                                nb01: (d_inner * 4) as u64,
                                nb02: (d_state * d_inner * 4) as u64,
                                nb03: (d_state * d_inner * 4) as u64,
                                nb10: 0,
                                nb11: (d_inner * 4) as u64,
                                nb12: (d_inner * n_tok * 4) as u64,
                                ns12: n_tok as u64,
                                nb13: (d_inner * n_tok * 4) as u64,
                                nb20: 0,
                                nb21: 4,
                                ns21: 1,
                                nb22: 4,
                                ne30: d_state as i64,
                                nb31: 4,
                                nb41: (d_state * 4) as u64,
                                nb42: (d_state * 4) as u64,
                                ns42: 1,
                                nb43: (d_state * 4) as u64,
                                nb51: (d_state * 4) as u64,
                                nb52: (d_state * 4) as u64,
                                ns52: 1,
                                nb53: (d_state * 4) as u64,
                                nb0: (n_head * d_inner * 4) as u64,
                            };

                            // Warmup llama.cpp
                            for _ in 0..10 {
                                run_llama_kernel(
                                    &queue,
                                    &llama_pipeline,
                                    &ll_x_buf,
                                    &ll_dt_buf,
                                    &ll_a_buf,
                                    &ll_b_buf,
                                    &ll_c_buf,
                                    &ll_output_buf,
                                    &args,
                                    threadgroup_mem,
                                    sgptg,
                                );
                            }

                            let mut llama_times = Vec::new();
                            for _ in 0..iterations {
                                let start = Instant::now();
                                run_llama_kernel(
                                    &queue,
                                    &llama_pipeline,
                                    &ll_x_buf,
                                    &ll_dt_buf,
                                    &ll_a_buf,
                                    &ll_b_buf,
                                    &ll_c_buf,
                                    &ll_output_buf,
                                    &args,
                                    threadgroup_mem,
                                    sgptg,
                                );
                                llama_times
                                    .push(start.elapsed().as_nanos() as f64);
                            }

                            let llama_mean_ns = llama_times.iter().sum::<f64>()
                                / iterations as f64;
                            let llama_mean_us = llama_mean_ns / 1000.0;

                            println!("  Time: {:.2} us", llama_mean_us);
                            println!(
                                "\nRatio (ours/llama.cpp): {:.2}x",
                                our_mean_us / llama_mean_us
                            );
                        }
                        Err(e) => println!("  Pipeline error: {}", e),
                    }
                }
                Err(e) => println!("  Function not found: {}", e),
            }
        }
        Err(e) => {
            println!("llama.cpp kernel compilation failed: {}", e);
            println!("Only our kernel benchmarked");
        }
    }
}

fn run_our_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    x_buf: &Buffer,
    dt_buf: &Buffer,
    a_buf: &Buffer,
    b_buf: &Buffer,
    c_buf: &Buffer,
    y_buf: &Buffer,
    s_buf: &Buffer,
    d_inner: usize,
    n_tok: usize,
) {
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();

    let sgptg = 4; // simdgroups per threadgroup
    let threadgroup_mem = (sgptg * N_SIMDWIDTH + sgptg * 2) * 4; // shared_sums + shared_x_dt + shared_dA

    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(s_buf), 0); // s0
    enc.set_buffer(1, Some(x_buf), 0); // x
    enc.set_buffer(2, Some(dt_buf), 0); // dt
    enc.set_buffer(3, Some(a_buf), 0); // A
    enc.set_buffer(4, Some(b_buf), 0); // B
    enc.set_buffer(5, Some(c_buf), 0); // C
    enc.set_buffer(6, Some(y_buf), 0); // dst (y + s_out)

    let ds = 16u32;
    let di = d_inner as u32;
    let nt = n_tok as u32;
    enc.set_bytes(7, 4, &ds as *const u32 as *const c_void);
    enc.set_bytes(8, 4, &di as *const u32 as *const c_void);
    enc.set_bytes(9, 4, &nt as *const u32 as *const c_void);
    enc.set_threadgroup_memory_length(0, threadgroup_mem as u64);

    // Grid: one threadgroup per (d_inner/32, head, sequence)
    let n_inner_groups = (d_inner + 31) / 32;
    let grid = MTLSize { width: n_inner_groups as u64, height: 1, depth: 1 };
    let tg =
        MTLSize { width: (sgptg * N_SIMDWIDTH) as u64, height: 1, depth: 1 };

    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

fn run_llama_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    x_buf: &Buffer,
    dt_buf: &Buffer,
    a_buf: &Buffer,
    b_buf: &Buffer,
    c_buf: &Buffer,
    output_buf: &Buffer,
    args: &LlamaArgs,
    threadgroup_mem: usize,
    sgptg: usize,
) {
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();

    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<LlamaArgs>() as u64,
        args as *const LlamaArgs as *const c_void,
    );
    enc.set_buffer(1, None, 0); // s0 - will use dst
    enc.set_buffer(2, Some(x_buf), 0);
    enc.set_buffer(3, Some(dt_buf), 0);
    enc.set_buffer(4, Some(a_buf), 0);
    enc.set_buffer(5, Some(b_buf), 0);
    enc.set_buffer(6, Some(c_buf), 0);
    enc.set_buffer(7, None, 0); // ids
    enc.set_buffer(8, Some(output_buf), 0);
    enc.set_threadgroup_memory_length(0, threadgroup_mem as u64);

    // Grid: one threadgroup per (d_inner_chunk, head, seq)
    let n_inner_groups = (args.d_inner as usize + 31) / 32; // i1 dimension
    let grid = MTLSize { width: n_inner_groups as u64, height: 1, depth: 1 };
    let tg =
        MTLSize { width: (sgptg * N_SIMDWIDTH) as u64, height: 1, depth: 1 };

    enc.dispatch_thread_groups(grid, tg);
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}
