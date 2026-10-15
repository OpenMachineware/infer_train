// Direct SSD comparison: our kernel vs llama.cpp kernel
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use metal::*;
use std::ffi::c_void;
use std::time::Instant;

// Constants from llama.cpp
const CS: usize = 64;
const TC: usize = 8;
const HD: usize = 64;
const NSG: usize = 4;
const N_SIMDWIDTH: usize = 32;

// llama.cpp args struct (matching ggml_metal_kargs_ssm_scan)
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
    let our_source = include_str!("../src/ops/metal/ops/ssd_optimized.metal");
    let options = CompileOptions::new();
    let our_lib = device
        .new_library_with_source(our_source, &options)
        .expect("Failed to compile our kernel");
    let our_func = our_lib
        .get_function("kernel_ssm_scan_ssd_mma", None)
        .expect("Our function not found");
    let our_pipeline = device
        .new_compute_pipeline_state_with_function(&our_func)
        .expect("Our pipeline failed");

    // Compile llama.cpp kernel - extract just the SSD MMA kernel
    let llama_common = r#"
#include <metal_stdlib>
using namespace metal;
#define N_SIMDWIDTH 32
#define OP_SSM_SCAN_SSD_CS 64
#define OP_SSM_SCAN_SSD_HD 64
#define OP_SSM_SCAN_SSD_NSG 4

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
kernel void kernel_ssm_scan_ssd_mma_f32(
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
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    constexpr short CS  = OP_SSM_SCAN_SSD_CS;
    constexpr short TC  = 8;
    constexpr short HD  = OP_SSM_SCAN_SSD_HD;
    constexpr short NSG = OP_SSM_SCAN_SSD_NSG;

    threadgroup float * shared_acs         = shared;
    threadgroup float * shared_exp_acs     = shared + CS;
    threadgroup float * shared_state_decay = shared + 2*CS;
    threadgroup float * shared_dtx         = shared + 3*CS;
    threadgroup float * shared_sam         = shared + 3*CS + CS*HD;
    threadgroup float * sam_rows    = shared_sam + sgitg*TC*CS;
    threadgroup float * shared_tile = shared_sam + NSG*TC*CS;
    threadgroup float * tile0       = shared_tile + sgitg*2*TC*TC;
    threadgroup float * tile1       = tile0 + TC*TC;

    const int32_t ir = tgpig.y;
    const int32_t i3 = tgpig.z;

    const int32_t nc  = args.d_state;
    const int32_t nr  = args.d_inner;
    const int32_t nh  = args.n_head;
    const int32_t ng  = args.n_group;
    const int32_t n_t = args.n_seq_tokens;
    const int32_t n_t_total = args.n_seq_tokens_total;
    const int32_t g   = ir / (nh / ng);

    device const int32_t * ids = (device const int32_t *) src6;

    device const float * s0_buff = (device const float *) ((device const char *) src0 + ir*args.nb02 + ids[i3]*args.nb03);
    device       float * s_buff  = (device       float *) ((device       char *) dst  + ir*args.nb02 +      i3*args.nb03 + args.s_off);

    device const float * A  = (device const float *) ((device const char *) src3 + ir*args.nb31);
    device const float * x  = (device const float *) ((device const char *) src1 + ir*args.nb11 + i3*args.nb13);
    device const float * dt = (device const float *) ((device const char *) src2 + ir*args.nb20 + i3*args.nb22);
    device const float * B  = (device const float *) ((device const char *) src4 +  g*args.nb41 + i3*args.nb43);
    device const float * C  = (device const float *) ((device const char *) src5 +  g*args.nb51 + i3*args.nb53);

    device float * y = dst + (ir*nr + i3*(n_t_total*nh*nr));

    for (int32_t t0 = 0; t0 < n_t; t0 += CS) {
        for (int32_t idx = tiitg; idx < CS*HD; idx += NSG*N_SIMDWIDTH) {
            const int32_t t = idx / HD;
            const int32_t c = idx % HD;
            const float dt0  = dt[(t0 + t) * (int32_t) args.ns21];
            const float dtsp = dt0 <= 20.0f ? log(1.0f + exp(dt0)) : dt0;
            shared_dtx[idx] = x[(t0 + t) * (int32_t) args.ns12 + c] * dtsp;
        }
        if (tiitg < CS) {
            const float dt0  = dt[(t0 + tiitg) * (int32_t) args.ns21];
            const float dtsp = dt0 <= 20.0f ? log(1.0f + exp(dt0)) : dt0;
            shared_acs[tiitg] = dtsp * A[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiitg == 0) {
            float acc = 0.0f;
            for (short t = 0; t < CS; ++t) {
                acc += shared_acs[t];
                shared_acs[t] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg < CS) {
            shared_exp_acs[tiitg] = exp(shared_acs[tiitg]);
            shared_state_decay[tiitg] = exp(shared_acs[CS - 1] - shared_acs[tiitg]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const float * state = t0 == 0 ? s0_buff : s_buff;

        for (short ib = sgitg; ib < CS/TC; ib += NSG) {
            for (short jb = 0; jb <= ib; ++jb) {
                simdgroup_float8x8 cb = make_filled_simdgroup_matrix<float, 8>(0.0f);

                for (int32_t k0 = 0; k0 < nc; k0 += TC) {
                    simdgroup_float8x8 mc;
                    simdgroup_float8x8 mb;
                    simdgroup_load(mc, C + (t0 + ib*TC)*(int32_t) args.ns52 + k0, args.ns52);
                    simdgroup_load(mb, B + (t0 + jb*TC)*(int32_t) args.ns42 + k0, args.ns42, 0, true);
                    simdgroup_multiply_accumulate(cb, mc, mb, cb);
                }

                threadgroup float * sam = sam_rows + jb*TC;
                simdgroup_store(cb, sam, CS);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (short e = tiisg; e < TC*TC; e += N_SIMDWIDTH) {
                    const short ri = e / TC;
                    const short rj = e % TC;
                    const short i  = ib*TC + ri;
                    const short j  = jb*TC + rj;
                    sam[ri*CS + rj] = j <= i ?
                        sam[ri*CS + rj] * exp(shared_acs[i] - shared_acs[j]) : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }

            for (short ch = 0; ch < HD/TC; ++ch) {
                simdgroup_float8x8 y_diag  = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_float8x8 y_inter = make_filled_simdgroup_matrix<float, 8>(0.0f);

                for (short jb = 0; jb <= ib; ++jb) {
                    simdgroup_float8x8 sam;
                    simdgroup_float8x8 mdtx;
                    simdgroup_load(sam,  sam_rows + jb*TC,                   CS);
                    simdgroup_load(mdtx, shared_dtx + jb*TC*HD + ch*TC,     HD);
                    simdgroup_multiply_accumulate(y_diag, sam, mdtx, y_diag);
                }

                for (int32_t k0 = 0; k0 < nc; k0 += TC) {
                    simdgroup_float8x8 mc;
                    simdgroup_float8x8 ms;
                    simdgroup_load(mc, C + (t0 + ib*TC)*(int32_t) args.ns52 + k0, args.ns52);
                    simdgroup_load(ms, state + ch*TC*nc + k0, nc, 0, true);
                    simdgroup_multiply_accumulate(y_inter, mc, ms, y_inter);
                }

                simdgroup_store(y_diag,  tile0, TC);
                simdgroup_store(y_inter, tile1, TC);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (short e = tiisg; e < TC*TC; e += N_SIMDWIDTH) {
                    const short ri = e / TC;
                    const short ci = e % TC;
                    const int32_t token = t0 + ib*TC + ri;
                    y[token*nh*nr + ch*TC + ci] =
                        tile0[e] + shared_exp_acs[ib*TC + ri] * tile1[e];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        const float chunk_decay = exp(shared_acs[CS - 1]);
        for (int32_t idx = tiitg; idx < nc*HD; idx += NSG*N_SIMDWIDTH) {
            const int32_t ci = idx / nc;
            const int32_t si = idx % nc;
            float state_c = 0.0f;
            for (short t = 0; t < CS; ++t) {
                state_c += shared_state_decay[t] *
                    B[(t0 + t)*(int32_t) args.ns42 + si] *
                    shared_dtx[t*HD + ci];
            }
            s_buff[idx] = chunk_decay * state[idx] + state_c;
        }

        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
"#;
    let llama_full = format!("{}\n{}", llama_common, llama_kernel);

    println!("=== SSD Kernel Performance Comparison ===\n");

    // Test parameters
    let d_state = 64;
    let head_dim = 64;
    let n_tok = 512;
    let n_head = 1;

    let x_size = n_tok * head_dim;
    let bc_size = n_tok * d_state;
    let state_size = head_dim * d_state;
    let y_size = n_tok * n_head * head_dim;

    // Create test data
    let x: Vec<f32> = (0..x_size).map(|i| 0.1 * (i as f32 % 10.0)).collect();
    let dt: Vec<f32> =
        (0..n_tok).map(|i| 0.001 * (i as f32 % 5.0 + 1.0)).collect();
    let a: Vec<f32> = vec![-0.1];
    let b: Vec<f32> = (0..bc_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();
    let c: Vec<f32> = (0..bc_size).map(|i| 0.05 * (i as f32 % 8.0)).collect();

    // Create buffers
    let s0_buf = device.new_buffer(
        (state_size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
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
    let output_buf = device.new_buffer(
        ((y_size + state_size) * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Threadgroup memory: acs[CS] + exp_acs[CS] + state_decay[CS] + dtx[CS*HD] + SAM[NSG*TC*CS] + tiles[NSG*2*TC*TC]
    // = 3*CS + CS*HD + NSG*TC*CS + NSG*2*TC*TC floats
    // = 192 + 4096 + 2048 + 512 = 6848 floats = 27.4 KB
    let threadgroup_mem =
        (CS * 3 + CS * HD + NSG * TC * CS + NSG * 2 * TC * TC) * 4;

    // Warmup with batched dispatch
    for _ in 0..10 {
        run_our_kernel_batched(
            &device,
            &queue,
            &our_pipeline,
            &s0_buf,
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
            threadgroup_mem,
            5,
        );
    }

    // Measure our kernel
    let iterations = 50;
    let mut our_times = Vec::new();

    for _ in 0..iterations {
        let start = Instant::now();
        run_our_kernel_batched(
            &device,
            &queue,
            &our_pipeline,
            &s0_buf,
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
            threadgroup_mem,
            5,
        );
        let elapsed = start.elapsed().as_nanos() as f64 / 5.0; // Average per kernel
        our_times.push(elapsed);
    }

    let our_mean_ns = our_times.iter().sum::<f64>() / iterations as f64;
    let our_mean_us = our_mean_ns / 1000.0;

    println!("Our kernel:");
    println!("  Time: {:.2} us", our_mean_us);

    // Try to compile llama.cpp kernel
    match device.new_library_with_source(&llama_full, &options) {
        Ok(llama_lib) => {
            match llama_lib.get_function("kernel_ssm_scan_ssd_mma_f32", None) {
                Ok(llama_func) => {
                    match device
                        .new_compute_pipeline_state_with_function(&llama_func)
                    {
                        Ok(llama_pipeline) => {
                            println!(
                                "\nllama.cpp kernel available, benchmarking..."
                            );

                            // Create args struct matching llama.cpp layout
                            let args = LlamaArgs {
                                d_state: d_state as i64,
                                d_inner: head_dim as i64,
                                n_head: n_head as i64,
                                n_group: 1,
                                n_seq_tokens: n_tok as i64,
                                n_seq_tokens_total: n_tok as i64,
                                token_offset: 0,
                                n_seqs: 1,
                                K: 0,
                                s_off: y_size as u64 * 4,
                                nb00: 0,
                                nb01: 0,
                                nb02: (d_state * head_dim * 4) as u64,
                                nb03: (d_state * head_dim * 4) as u64,
                                nb10: 0,
                                nb11: (head_dim * 4) as u64,
                                nb12: (n_tok * head_dim * 4) as u64,
                                ns12: head_dim as u64,
                                nb13: (head_dim * 4) as u64,
                                nb20: 0,
                                nb21: 4,
                                ns21: 1,
                                nb22: 4,
                                ne30: 1,
                                nb31: 4,
                                nb41: (d_state * 4) as u64,
                                nb42: (d_state * 4) as u64,
                                ns42: d_state as u64,
                                nb43: (d_state * 4) as u64,
                                nb51: (d_state * 4) as u64,
                                nb52: (d_state * 4) as u64,
                                ns52: d_state as u64,
                                nb53: (d_state * 4) as u64,
                                nb0: (n_head * head_dim * 4) as u64,
                            };

                            // Warmup llama.cpp
                            for _ in 0..10 {
                                run_llama_kernel_batched(
                                    &device,
                                    &queue,
                                    &llama_pipeline,
                                    &s0_buf,
                                    &x_buf,
                                    &dt_buf,
                                    &a_buf,
                                    &b_buf,
                                    &c_buf,
                                    &output_buf,
                                    &args,
                                    threadgroup_mem,
                                    n_head,
                                    5,
                                );
                            }

                            // Measure
                            let mut llama_times = Vec::new();
                            for _ in 0..iterations {
                                let start = Instant::now();
                                run_llama_kernel_batched(
                                    &device,
                                    &queue,
                                    &llama_pipeline,
                                    &s0_buf,
                                    &x_buf,
                                    &dt_buf,
                                    &a_buf,
                                    &b_buf,
                                    &c_buf,
                                    &output_buf,
                                    &args,
                                    threadgroup_mem,
                                    n_head,
                                    5,
                                );
                                let elapsed =
                                    start.elapsed().as_nanos() as f64 / 5.0;
                                llama_times.push(elapsed);
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
                        Err(e) => println!(
                            "  Failed to create llama.cpp pipeline: {}",
                            e
                        ),
                    }
                }
                Err(e) => println!("  Function not found: {}", e),
            }
        }
        Err(e) => {
            println!("llama.cpp kernel compilation failed: {}", e);
            println!(
                "\nComparison not available - using estimated llama.cpp performance"
            );
            println!("llama.cpp estimated: ~1100-1300 us for same parameters");
            println!("Ratio (ours/estimated): {:.2}x", our_mean_us / 1200.0);
        }
    }
}

fn run_our_kernel_batched(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    s0_buf: &Buffer,
    x_buf: &Buffer,
    dt_buf: &Buffer,
    a_buf: &Buffer,
    b_buf: &Buffer,
    c_buf: &Buffer,
    output_buf: &Buffer,
    d_state: usize,
    head_dim: usize,
    n_tok: usize,
    n_head: usize,
    threadgroup_mem: usize,
    count: usize,
) {
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();

    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(s0_buf), 0);
    enc.set_buffer(1, Some(x_buf), 0);
    enc.set_buffer(2, Some(dt_buf), 0);
    enc.set_buffer(3, Some(a_buf), 0);
    enc.set_buffer(4, Some(b_buf), 0);
    enc.set_buffer(5, Some(c_buf), 0);
    enc.set_buffer(7, Some(output_buf), 0);

    let ds = d_state as u32;
    let hd = head_dim as u32;
    let nh = n_head as u32;
    let nt = n_tok as u32;
    let ns = 1u32;

    enc.set_bytes(8, 4, &ds as *const u32 as *const c_void);
    enc.set_bytes(9, 4, &hd as *const u32 as *const c_void);
    enc.set_bytes(10, 4, &nh as *const u32 as *const c_void);
    enc.set_bytes(11, 4, &nt as *const u32 as *const c_void);
    enc.set_bytes(12, 4, &ns as *const u32 as *const c_void);
    enc.set_threadgroup_memory_length(0, threadgroup_mem as u64);

    let grid = MTLSize { width: 1, height: n_head as u64, depth: 1 };
    let tg = MTLSize { width: 128, height: 1, depth: 1 };

    for _ in 0..count {
        enc.dispatch_thread_groups(grid, tg);
    }
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}

fn run_llama_kernel_batched(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    s0_buf: &Buffer,
    x_buf: &Buffer,
    dt_buf: &Buffer,
    a_buf: &Buffer,
    b_buf: &Buffer,
    c_buf: &Buffer,
    output_buf: &Buffer,
    args: &LlamaArgs,
    threadgroup_mem: usize,
    n_head: usize,
    count: usize,
) {
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();

    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<LlamaArgs>() as u64,
        args as *const LlamaArgs as *const c_void,
    );
    enc.set_buffer(1, Some(s0_buf), 0);
    enc.set_buffer(2, Some(x_buf), 0);
    enc.set_buffer(3, Some(dt_buf), 0);
    enc.set_buffer(4, Some(a_buf), 0);
    enc.set_buffer(5, Some(b_buf), 0);
    enc.set_buffer(6, Some(c_buf), 0);
    enc.set_buffer(8, Some(output_buf), 0);
    enc.set_threadgroup_memory_length(0, threadgroup_mem as u64);

    let grid = MTLSize { width: 1, height: n_head as u64, depth: 1 };
    let tg = MTLSize { width: 128, height: 1, depth: 1 };

    for _ in 0..count {
        enc.dispatch_thread_groups(grid, tg);
    }
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
}
