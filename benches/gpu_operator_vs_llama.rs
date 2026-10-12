// GPU Operator Benchmark vs llama.cpp
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use metal::{Device, MTLSize};
use std::time::Instant;

const LLAMA_COMMON_HEADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(a, b) { auto tmp = (a); (a) = (b); (b) = tmp; }
enum ggml_sort_order { GGML_SORT_ORDER_ASC = 0, GGML_SORT_ORDER_DESC = 1 };
typedef struct { int32_t k0; int32_t k1; int32_t s0; int32_t s1; int32_t p0; int32_t p1; int64_t IH; int64_t IW; int64_t OH; int64_t OW; int64_t np; } ggml_metal_kargs_pool_2d;
typedef struct { int32_t k0; int32_t s0; int32_t p0; int64_t IW; int64_t OW; int64_t np; } ggml_metal_kargs_pool_1d;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; int32_t ne03; uint64_t nb01; uint64_t nb02; uint64_t nb03; int32_t top_k; } ggml_metal_kargs_top_k;
typedef struct { int64_t ne00; int64_t ne01; int64_t ne02; int64_t ne03; int64_t nb01; int64_t nb02; int64_t nb03; int64_t ne0; int64_t top_k; } ggml_metal_kargs_argsort;
"#;

const LLAMA_TOP_K_KERNEL: &str = r#"
static uint ggml_top_k_f2ui(float x) {
    uint y = as_type<uint>(x);
    if ((y & 0x80000000u) != 0u) { y ^= 0xFFFFFFFFu; }
    else { y |= 0x80000000u; }
    return y;
}

kernel void kernel_top_k_f32_i32(
    constant ggml_metal_kargs_top_k& args [[buffer(0)]],
    device const char* src0 [[buffer(1)]],
    device int32_t* dst [[buffer(2)]],
    threadgroup atomic_uint* histogram [[threadgroup(0)]],
    threadgroup uint* sh_bucket [[threadgroup(1)]],
    threadgroup uint* sh_above [[threadgroup(2)]],
    threadgroup atomic_uint* out_count [[threadgroup(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]]) {
    const uint ncols = args.ne00;
    const uint top_k = args.top_k;
    const uint i01 = tgpig[0];
    const uint i02 = tgpig[1];
    const uint i03 = tgpig[2];

    device const float* src0_row = (device const float*)(src0 + args.nb01 * i01 + args.nb02 * i02 + args.nb03 * i03);
    device int32_t* dst_row = dst + top_k * (i01 + args.ne01 * i02 + args.ne01 * args.ne02 * i03);

    const uint tid = tpitg.x;
    const uint ntg_x = ntg.x;
    uint prefix = 0;
    uint desired = top_k;

    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint i = tid; i < 256; i += ntg_x) {
            atomic_store_explicit(&histogram[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint hi_mask = (shift + 8 >= 32) ? 0u : (0xFFFFFFFFu << uint(shift + 8));
        const uint prefix_hi = prefix & hi_mask;

        for (uint i = tid; i < ncols; i += ntg_x) {
            const uint key = ggml_top_k_f2ui(src0_row[i]);
            if ((key & hi_mask) == prefix_hi) {
                atomic_fetch_add_explicit(&histogram[(key >> uint(shift)) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            uint acc = 0;
            uint b = 0;
            for (int bb = 255; bb >= 0; --bb) {
                const uint c = atomic_load_explicit(&histogram[bb], memory_order_relaxed);
                if (acc + c >= desired) { b = uint(bb); break; }
                acc += c;
            }
            *sh_bucket = b;
            *sh_above = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        prefix |= *sh_bucket << uint(shift);
        desired -= *sh_above;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) { atomic_store_explicit(out_count, 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint threshold = prefix;
    for (uint i = tid; i < ncols; i += ntg_x) {
        if (ggml_top_k_f2ui(src0_row[i]) > threshold) {
            const uint pos = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
            dst_row[pos] = (int32_t)i;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < ncols; i += ntg_x) {
        if (ggml_top_k_f2ui(src0_row[i]) == threshold) {
            const uint pos = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
            if (pos < top_k) { dst_row[pos] = (int32_t)i; }
        }
    }
}
"#;

const LLAMA_ARGSORT_KERNEL: &str = r#"
template<ggml_sort_order order>
kernel void kernel_argsort_f32_i32(
    constant ggml_metal_kargs_argsort& args,
    device const char* src0,
    device int32_t* dst,
    threadgroup int32_t* shmem_i32 [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort3 tpitg [[thread_position_in_threadgroup]],
    ushort3 ntg [[threads_per_threadgroup]]) {
    const int col = tpitg[0];
    const int ib = tgpig[0] / args.ne01;

    const int i00 = ib * ntg.x;
    const int i01 = tgpig[0] % args.ne01;
    const int i02 = tgpig[1];
    const int i03 = tgpig[2];

    device const float* src0_row = (device const float*)(src0 + args.nb01 * i01 + args.nb02 * i02 + args.nb03 * i03);

    shmem_i32[col] = i00 + col;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= int(ntg.x); k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= args.ne00 ||
                        (shmem_i32[ixj] < args.ne00 && (order == GGML_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]]))) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                } else {
                    if (shmem_i32[ixj] >= args.ne00 ||
                        (shmem_i32[col] < args.ne00 && (order == GGML_SORT_ORDER_ASC ?
                            src0_row[shmem_i32[col]] < src0_row[shmem_i32[ixj]] :
                            src0_row[shmem_i32[col]] > src0_row[shmem_i32[ixj]]))) {
                        SWAP(shmem_i32[col], shmem_i32[ixj]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const int64_t i0 = ib * args.top_k;
    if (i0 + col < args.ne0 && col < args.top_k) {
        dst += i0 + args.ne0 * i01 + args.ne0 * args.ne01 * i02 + args.ne0 * args.ne01 * args.ne02 * i03;
        dst[col] = shmem_i32[col];
    }
}

template [[host_name("kernel_argsort_f32_i32_desc")]] kernel void kernel_argsort_f32_i32<GGML_SORT_ORDER_DESC>(
    constant ggml_metal_kargs_argsort& args,
    device const char* src0,
    device int32_t* dst,
    threadgroup int32_t* shmem_i32,
    uint3 tgpig,
    ushort3 tpitg,
    ushort3 ntg);
"#;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("=== GPU Operator Benchmark ===");
    println!("Method: Alternating A-B test where llama.cpp kernel available");
    println!();

    bench_pool_2d(&device, &queue);
    bench_top_k(&device, &queue);
    bench_argsort(&device, &queue);
}

fn bench_pool_2d(device: &Device, queue: &metal::CommandQueue) {
    println!("========================================");
    println!("POOL_2D (Max Pooling) vs llama.cpp");
    println!("========================================");

    let self_source = include_str!("../src/ops/metal/pool_2d.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let self_lib = device
        .new_library_with_source(self_source, &options)
        .expect("Self compile failed");
    let self_kernel =
        self_lib.get_function("pool_2d_max_f32", None).expect("No self kernel");
    let self_pipeline = device
        .new_compute_pipeline_state_with_function(&self_kernel)
        .expect("No self pipeline");

    let llama_pool_source = include_str!(
        "../../llama.cpp-0.6.0/ggml/src/ggml-metal/kernels/pool.metal"
    )
    .replace("#include \"common.h\"", "");
    let llama_source =
        format!("{}\n{}", LLAMA_COMMON_HEADER, llama_pool_source);
    let llama_lib = device
        .new_library_with_source(&llama_source, &options)
        .expect("llama compile failed");
    let llama_kernel = llama_lib
        .get_function("kernel_pool_2d_max_f32", None)
        .expect("No llama kernel");
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_kernel)
        .expect("No llama pipeline");

    let configs = [
        (64, 64, 64, 2, 2, 2, 2, 0, 0),
        (64, 128, 128, 2, 2, 2, 2, 0, 0),
        (64, 256, 256, 2, 2, 2, 2, 0, 0),
        (64, 512, 512, 2, 2, 2, 2, 0, 0),
        (128, 128, 128, 3, 3, 2, 2, 1, 1),
        (64, 256, 256, 4, 4, 4, 4, 0, 0),
        (64, 256, 256, 5, 5, 1, 1, 2, 2),
    ];

    for (ic, ih, iw, kh, kw, sh, sw, ph, pw) in configs {
        let oh = (ih - kh + 2 * ph) / sh + 1;
        let ow = (iw - kw + 2 * pw) / sw + 1;
        let input_size = (ic * ih * iw) as usize;
        let output_size = (ic * oh * ow) as usize;
        let np = output_size as i64;

        let input_data: Vec<f32> =
            (0..input_size).map(|i| (i % 1000) as f32 / 100.0).collect();
        let input_buf = device.new_buffer(
            (input_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_self = device.new_buffer(
            (output_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_llama = device.new_buffer(
            (output_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                input_data.as_ptr(),
                input_buf.contents() as *mut f32,
                input_size,
            );
        }

        // Warmup
        for _ in 0..10 {
            run_pool_self(
                &self_pipeline,
                queue,
                &input_buf,
                &output_self,
                ic,
                ih,
                iw,
                oh,
                ow,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                np,
            );
            run_pool_llama(
                &llama_pipeline,
                queue,
                &input_buf,
                &output_llama,
                ic,
                ih,
                iw,
                oh,
                ow,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                np,
            );
        }

        // Alternating benchmark
        let rounds = 500;
        let mut self_times = Vec::with_capacity(rounds);
        let mut llama_times = Vec::with_capacity(rounds);

        for i in 0..rounds {
            if i % 2 == 0 {
                let t = Instant::now();
                run_pool_self(
                    &self_pipeline,
                    queue,
                    &input_buf,
                    &output_self,
                    ic,
                    ih,
                    iw,
                    oh,
                    ow,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    np,
                );
                self_times.push(t.elapsed().as_nanos() as f64);
                let t = Instant::now();
                run_pool_llama(
                    &llama_pipeline,
                    queue,
                    &input_buf,
                    &output_llama,
                    ic,
                    ih,
                    iw,
                    oh,
                    ow,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    np,
                );
                llama_times.push(t.elapsed().as_nanos() as f64);
            } else {
                let t = Instant::now();
                run_pool_llama(
                    &llama_pipeline,
                    queue,
                    &input_buf,
                    &output_llama,
                    ic,
                    ih,
                    iw,
                    oh,
                    ow,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    np,
                );
                llama_times.push(t.elapsed().as_nanos() as f64);
                let t = Instant::now();
                run_pool_self(
                    &self_pipeline,
                    queue,
                    &input_buf,
                    &output_self,
                    ic,
                    ih,
                    iw,
                    oh,
                    ow,
                    kh,
                    kw,
                    sh,
                    sw,
                    ph,
                    pw,
                    np,
                );
                self_times.push(t.elapsed().as_nanos() as f64);
            }
        }

        let avg_self: f64 = self_times.iter().sum::<f64>() / rounds as f64;
        let avg_llama: f64 = llama_times.iter().sum::<f64>() / rounds as f64;
        let ratio = avg_llama / avg_self * 100.0;

        println!(
            "{}ch {}x{} -> {}x{}, {}x{} s{}: Self {:6.1}us | llama {:6.1}us | {:5.1}%",
            ic, ih, iw, oh, ow, kh, kw, sh, avg_self / 1000.0, avg_llama / 1000.0, ratio
        );
    }
    println!();
}

#[repr(C)]
struct Pool2DArgs {
    k0: i32,
    k1: i32,
    s0: i32,
    s1: i32,
    p0: i32,
    p1: i32,
    IH: i64,
    IW: i64,
    OH: i64,
    OW: i64,
    np: i64,
}

fn run_pool_self(
    pipeline: &metal::ComputePipelineState,
    queue: &metal::CommandQueue,
    input: &metal::Buffer,
    output: &metal::Buffer,
    ic: i32,
    ih: i32,
    iw: i32,
    oh: i32,
    ow: i32,
    kh: i32,
    kw: i32,
    sh: i32,
    sw: i32,
    ph: i32,
    pw: i32,
    np: i64,
) {
    let args = Pool2DArgs {
        k0: kw,
        k1: kh,
        s0: sw,
        s1: sh,
        p0: pw,
        p1: ph,
        IH: ih as i64,
        IW: iw as i64,
        OH: oh as i64,
        OW: ow as i64,
        np,
    };

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<Pool2DArgs>() as u64,
        &args as *const _ as *const _,
    );
    enc.set_buffer(1, Some(input), 0);
    enc.set_buffer(2, Some(output), 0);
    let tg = 64u64;
    let groups = ((np as u64 + tg - 1) / tg).max(1);
    enc.dispatch_thread_groups(
        MTLSize { width: groups, height: 1, depth: 1 },
        MTLSize { width: tg, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn run_pool_llama(
    pipeline: &metal::ComputePipelineState,
    queue: &metal::CommandQueue,
    input: &metal::Buffer,
    output: &metal::Buffer,
    ic: i32,
    ih: i32,
    iw: i32,
    oh: i32,
    ow: i32,
    kh: i32,
    kw: i32,
    sh: i32,
    sw: i32,
    ph: i32,
    pw: i32,
    np: i64,
) {
    let args = Pool2DArgs {
        k0: kw,
        k1: kh,
        s0: sw,
        s1: sh,
        p0: pw,
        p1: ph,
        IH: ih as i64,
        IW: iw as i64,
        OH: oh as i64,
        OW: ow as i64,
        np,
    };

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(input), 0);
    enc.set_buffer(1, Some(output), 0);
    enc.set_bytes(
        2,
        std::mem::size_of::<Pool2DArgs>() as u64,
        &args as *const _ as *const _,
    );
    let tg = 64u64;
    let groups = ((np as u64 + tg - 1) / tg).max(1);
    enc.dispatch_thread_groups(
        MTLSize { width: groups, height: 1, depth: 1 },
        MTLSize { width: tg, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

// ============================================================================
// TOP_K benchmark vs llama.cpp
// ============================================================================

fn bench_top_k(device: &Device, queue: &metal::CommandQueue) {
    println!("========================================");
    println!("TOP_K vs llama.cpp (histogram-based radix selection)");
    println!("========================================");

    let self_source = include_str!("../src/ops/metal/top_k.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let self_lib = device
        .new_library_with_source(self_source, &options)
        .expect("Self compile failed");
    let self_kernel =
        self_lib.get_function("top_k_f32_i32", None).expect("No self kernel");
    let self_pipeline = device
        .new_compute_pipeline_state_with_function(&self_kernel)
        .expect("No self pipeline");

    let llama_source =
        format!("{}\n{}", LLAMA_COMMON_HEADER, LLAMA_TOP_K_KERNEL);
    let llama_lib = device
        .new_library_with_source(&llama_source, &options)
        .expect("llama compile failed");
    let llama_kernel = llama_lib
        .get_function("kernel_top_k_f32_i32", None)
        .expect("No llama kernel");
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_kernel)
        .expect("No llama pipeline");

    // Test different sizes: (n, k)
    let configs = [
        (256, 8),
        (256, 16),
        (256, 32),
        (1024, 16),
        (1024, 32),
        (1024, 64),
        (4096, 32),
        (4096, 64),
        (4096, 128),
        (8192, 64),
        (8192, 128),
        (8192, 256),
    ];

    for (n, k) in configs {
        let input_data: Vec<f32> =
            (0..n as usize).map(|i| ((i * 7) % 1000) as f32 / 10.0).collect();
        let input_buf = device.new_buffer(
            (n as usize * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_self = device.new_buffer(
            (k as usize * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_llama = device.new_buffer(
            (k as usize * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                input_data.as_ptr(),
                input_buf.contents() as *mut f32,
                n as usize,
            );
        }

        // Warmup
        for _ in 0..10 {
            run_top_k_self(
                &self_pipeline,
                queue,
                &input_buf,
                &output_self,
                n,
                k,
            );
            run_top_k_llama(
                &llama_pipeline,
                queue,
                &input_buf,
                &output_llama,
                n,
                k,
            );
        }

        // Alternating benchmark
        let rounds = 500;
        let mut self_times = Vec::with_capacity(rounds);
        let mut llama_times = Vec::with_capacity(rounds);

        for i in 0..rounds {
            if i % 2 == 0 {
                let t = Instant::now();
                run_top_k_self(
                    &self_pipeline,
                    queue,
                    &input_buf,
                    &output_self,
                    n,
                    k,
                );
                self_times.push(t.elapsed().as_nanos() as f64);
                let t = Instant::now();
                run_top_k_llama(
                    &llama_pipeline,
                    queue,
                    &input_buf,
                    &output_llama,
                    n,
                    k,
                );
                llama_times.push(t.elapsed().as_nanos() as f64);
            } else {
                let t = Instant::now();
                run_top_k_llama(
                    &llama_pipeline,
                    queue,
                    &input_buf,
                    &output_llama,
                    n,
                    k,
                );
                llama_times.push(t.elapsed().as_nanos() as f64);
                let t = Instant::now();
                run_top_k_self(
                    &self_pipeline,
                    queue,
                    &input_buf,
                    &output_self,
                    n,
                    k,
                );
                self_times.push(t.elapsed().as_nanos() as f64);
            }
        }

        let avg_self: f64 = self_times.iter().sum::<f64>() / rounds as f64;
        let avg_llama: f64 = llama_times.iter().sum::<f64>() / rounds as f64;
        let ratio = avg_llama / avg_self * 100.0;

        println!(
            "n={:5}, k={:3}: Self {:6.1}us | llama {:6.1}us | {:5.1}%",
            n,
            k,
            avg_self / 1000.0,
            avg_llama / 1000.0,
            ratio
        );
    }
    println!();
}

#[repr(C)]
struct TopKArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    top_k: i32,
}

fn run_top_k_self(
    pipeline: &metal::ComputePipelineState,
    queue: &metal::CommandQueue,
    input: &metal::Buffer,
    output: &metal::Buffer,
    n: i32,
    k: i32,
) {
    let args = TopKArgs {
        ne00: n,
        ne01: 1,
        ne02: 1,
        ne03: 1,
        nb01: (n * 4) as u64,
        nb02: (n * 4) as u64,
        nb03: (n * 4) as u64,
        top_k: k,
    };

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<TopKArgs>() as u64,
        &args as *const _ as *const _,
    );
    enc.set_buffer(1, Some(input), 0);
    enc.set_buffer(2, Some(output), 0);
    enc.set_threadgroup_memory_length(0, 1024);
    enc.set_threadgroup_memory_length(1, 4);
    enc.set_threadgroup_memory_length(2, 4);
    enc.set_threadgroup_memory_length(3, 4);
    enc.dispatch_thread_groups(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn run_top_k_llama(
    pipeline: &metal::ComputePipelineState,
    queue: &metal::CommandQueue,
    input: &metal::Buffer,
    output: &metal::Buffer,
    n: i32,
    k: i32,
) {
    let args = TopKArgs {
        ne00: n,
        ne01: 1,
        ne02: 1,
        ne03: 1,
        nb01: (n * 4) as u64,
        nb02: (n * 4) as u64,
        nb03: (n * 4) as u64,
        top_k: k,
    };

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<TopKArgs>() as u64,
        &args as *const _ as *const _,
    );
    enc.set_buffer(1, Some(input), 0);
    enc.set_buffer(2, Some(output), 0);
    enc.set_threadgroup_memory_length(0, 1024);
    enc.set_threadgroup_memory_length(1, 4);
    enc.set_threadgroup_memory_length(2, 4);
    enc.set_threadgroup_memory_length(3, 4);
    enc.dispatch_thread_groups(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 32, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

// ============================================================================
// ARGSORT benchmark vs llama.cpp
// ============================================================================

fn bench_argsort(device: &Device, queue: &metal::CommandQueue) {
    println!("========================================");
    println!("ARGSORT vs llama.cpp (Bitonic Sort)");
    println!("========================================");

    let self_source = include_str!("../src/ops/metal/argsort.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let self_lib = device
        .new_library_with_source(self_source, &options)
        .expect("Self compile failed");
    let self_kernel = self_lib
        .get_function("kernel_argsort_f32_i32_desc", None)
        .expect("No self kernel");
    let self_pipeline = device
        .new_compute_pipeline_state_with_function(&self_kernel)
        .expect("No self pipeline");

    let llama_source =
        format!("{}\n{}", LLAMA_COMMON_HEADER, LLAMA_ARGSORT_KERNEL);
    let llama_lib = device
        .new_library_with_source(&llama_source, &options)
        .expect("llama compile failed");
    let llama_kernel = llama_lib
        .get_function("kernel_argsort_f32_i32_desc", None)
        .expect("No llama kernel");
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_kernel)
        .expect("No llama pipeline");

    // Test different sizes
    let configs = [256, 512, 1024, 2048];

    for n in configs {
        let input_data: Vec<f32> =
            (0..n as usize).map(|i| ((i * 7) % 1000) as f32 / 10.0).collect();
        let input_buf = device.new_buffer(
            (n as usize * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_self = device.new_buffer(
            (n as usize * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_llama = device.new_buffer(
            (n as usize * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                input_data.as_ptr(),
                input_buf.contents() as *mut f32,
                n as usize,
            );
        }

        let thread_count = (n as u64).next_power_of_two();

        // Warmup
        for _ in 0..10 {
            run_argsort_template(
                &self_pipeline,
                queue,
                &input_buf,
                &output_self,
                n,
                n,
                thread_count,
            );
            run_argsort_template(
                &llama_pipeline,
                queue,
                &input_buf,
                &output_llama,
                n,
                n,
                thread_count,
            );
        }

        // Alternating benchmark
        let rounds = 500;
        let mut self_times = Vec::with_capacity(rounds);
        let mut llama_times = Vec::with_capacity(rounds);

        for i in 0..rounds {
            if i % 2 == 0 {
                let t = Instant::now();
                run_argsort_template(
                    &self_pipeline,
                    queue,
                    &input_buf,
                    &output_self,
                    n,
                    n,
                    thread_count,
                );
                self_times.push(t.elapsed().as_nanos() as f64);
                let t = Instant::now();
                run_argsort_template(
                    &llama_pipeline,
                    queue,
                    &input_buf,
                    &output_llama,
                    n,
                    n,
                    thread_count,
                );
                llama_times.push(t.elapsed().as_nanos() as f64);
            } else {
                let t = Instant::now();
                run_argsort_template(
                    &llama_pipeline,
                    queue,
                    &input_buf,
                    &output_llama,
                    n,
                    n,
                    thread_count,
                );
                llama_times.push(t.elapsed().as_nanos() as f64);
                let t = Instant::now();
                run_argsort_template(
                    &self_pipeline,
                    queue,
                    &input_buf,
                    &output_self,
                    n,
                    n,
                    thread_count,
                );
                self_times.push(t.elapsed().as_nanos() as f64);
            }
        }

        let avg_self: f64 = self_times.iter().sum::<f64>() / rounds as f64;
        let avg_llama: f64 = llama_times.iter().sum::<f64>() / rounds as f64;
        let ratio = avg_llama / avg_self * 100.0;

        println!(
            "n={:4}: Self {:6.1}us | llama {:6.1}us | {:5.1}%",
            n,
            avg_self / 1000.0,
            avg_llama / 1000.0,
            ratio
        );
    }
    println!();
}

#[repr(C)]
struct ArgsortArgs {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb01: i64,
    nb02: i64,
    nb03: i64,
    ne0: i64,
    top_k: i64,
}

fn run_argsort_template(
    pipeline: &metal::ComputePipelineState,
    queue: &metal::CommandQueue,
    input: &metal::Buffer,
    output: &metal::Buffer,
    n: i32,
    top_k: i32,
    thread_count: u64,
) {
    let args = ArgsortArgs {
        ne00: n as i64,
        ne01: 1,
        ne02: 1,
        ne03: 1,
        nb01: (n * 4) as i64,
        nb02: (n * 4) as i64,
        nb03: (n * 4) as i64,
        ne0: top_k as i64,
        top_k: top_k as i64,
    };

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<ArgsortArgs>() as u64,
        &args as *const _ as *const _,
    );
    enc.set_buffer(1, Some(input), 0);
    enc.set_buffer(2, Some(output), 0);
    enc.set_threadgroup_memory_length(0, (thread_count as usize * 4) as u64);
    enc.dispatch_thread_groups(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: thread_count, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}
