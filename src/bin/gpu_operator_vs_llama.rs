// GPU Operator Benchmark vs llama.cpp (where possible)
// SPDX-FileCopyrightText: Copyright (c) 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

use metal::{Device, MTLSize};
use std::time::Instant;

const LLAMA_COMMON_HEADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(a, b) { int32_t tmp = (a); (a) = (b); (b) = tmp; }
#define GGML_SORT_ORDER_ASC  0
#define GGML_SORT_ORDER_DESC 1
typedef struct { int32_t k0; int32_t k1; int32_t s0; int32_t s1; int32_t p0; int32_t p1; int64_t IH; int64_t IW; int64_t OH; int64_t OW; int64_t np; } ggml_metal_kargs_pool_2d;
typedef struct { int32_t k0; int32_t s0; int32_t p0; int64_t IW; int64_t OW; int64_t np; } ggml_metal_kargs_pool_1d;
typedef struct { int64_t ne00; int64_t ne01; int64_t ne02; int64_t ne03; int64_t nb01; int64_t nb02; int64_t nb03; int64_t top_k; } ggml_metal_kargs_top_k;
typedef struct { int64_t ne00; int64_t ne01; int64_t ne02; int64_t ne03; int64_t nb01; int64_t nb02; int64_t nb03; int64_t ne0; int64_t top_k; } ggml_metal_kargs_argsort;
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
    bench_conv_2d(&device, &queue);
}

fn bench_pool_2d(device: &Device, queue: &metal::CommandQueue) {
    println!("========================================");
    println!("POOL_2D (Max Pooling) vs llama.cpp");
    println!("========================================");

    let self_source = include_str!("../../shaders/pool_2d.metal");
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
        "../../llama.cpp-0.5.0/ggml/src/ggml-metal/kernels/pool.metal"
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
        (64, 1024, 1024, 2, 2, 2, 2, 0, 0), // Larger
        (128, 128, 128, 3, 3, 2, 2, 1, 1),
        (64, 256, 256, 4, 4, 4, 4, 0, 0),
        (64, 256, 256, 5, 5, 1, 1, 2, 2), // Heavy compute
        (64, 256, 256, 7, 7, 1, 1, 3, 3), // Heavier
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

        for _ in 0..rounds {
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
        }

        let avg_self: f64 = self_times.iter().sum::<f64>() / rounds as f64;
        let avg_llama: f64 = llama_times.iter().sum::<f64>() / rounds as f64;
        let ratio = avg_llama / avg_self * 100.0;
        let throughput = (output_size as f64 * 4.0) / (avg_self * 1e-9) / 1e9;

        println!("{}ch {}x{} -> {}x{}, {}x{} s{}: Self {:6.1}us ({:.1}GB/s) | llama {:6.1}us | {:5.1}%",
                 ic, ih, iw, oh, ow, kh, kw, sh, avg_self/1000.0, throughput, avg_llama/1000.0, ratio);
    }
    println!();
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
) {
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
    let np = (ic as i64) * (oh as i64) * (ow as i64);
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
        &args as *const Pool2DArgs as *const std::ffi::c_void,
    );
    enc.set_buffer(1, Some(input), 0);
    enc.set_buffer(2, Some(output), 0);
    let threadgroup_size = 64u64;
    let num_groups =
        ((np as u64 + threadgroup_size - 1) / threadgroup_size).max(1);
    enc.dispatch_thread_groups(
        MTLSize { width: num_groups, height: 1, depth: 1 },
        MTLSize { width: threadgroup_size, height: 1, depth: 1 },
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
        &args as *const Pool2DArgs as *const std::ffi::c_void,
    );
    let threadgroup_size = 64u64;
    let num_groups =
        ((np as u64 + threadgroup_size - 1) / threadgroup_size).max(1);
    enc.dispatch_thread_groups(
        MTLSize { width: num_groups, height: 1, depth: 1 },
        MTLSize { width: threadgroup_size, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn bench_top_k(device: &Device, queue: &metal::CommandQueue) {
    println!("========================================");
    println!("TOP_K (Self-written only - histogram-based)");
    println!("========================================");

    let source = include_str!("../../shaders/top_k.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let lib = device
        .new_library_with_source(source, &options)
        .expect("Compile failed");
    let kernel = lib.get_function("top_k_f32_i32", None).expect("No kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("No pipeline");

    let configs = [(256, 8), (1024, 16), (4096, 32), (8192, 64)];

    for (n, k) in configs {
        let input_data: Vec<f32> =
            (0..n as usize).map(|i| ((i * 7) % 1000) as f32 / 10.0).collect();
        let input_buf = device.new_buffer(
            (n as usize * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buf = device.new_buffer(
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
            run_top_k(&pipeline, queue, &input_buf, &output_buf, n, k);
        }

        let rounds = 500;
        let mut times = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let t = Instant::now();
            run_top_k(&pipeline, queue, &input_buf, &output_buf, n, k);
            times.push(t.elapsed().as_nanos() as f64);
        }

        let avg = times.iter().sum::<f64>() / rounds as f64;
        println!("n={:5}, k={:3}: {:8.2} us", n, k, avg / 1000.0);
    }
    println!();
}

fn run_top_k(
    pipeline: &metal::ComputePipelineState,
    queue: &metal::CommandQueue,
    input: &metal::Buffer,
    output: &metal::Buffer,
    n: i32,
    k: i32,
) {
    #[repr(C)]
    struct TopKArgs {
        ne00: i64,
        ne01: i64,
        ne02: i64,
        ne03: i64,
        nb01: i64,
        nb02: i64,
        nb03: i64,
        top_k: i64,
    }
    let args = TopKArgs {
        ne00: n as i64,
        ne01: 1,
        ne02: 1,
        ne03: 1,
        nb01: (n * 4) as i64,
        nb02: (n * 4) as i64,
        nb03: (n * 4) as i64,
        top_k: k as i64,
    };

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<TopKArgs>() as u64,
        &args as *const TopKArgs as *const std::ffi::c_void,
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

fn bench_argsort(device: &Device, queue: &metal::CommandQueue) {
    println!("========================================");
    println!("ARGSORT (Self-written only - Bitonic Sort)");
    println!("========================================");

    let source = include_str!("../../shaders/argsort.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let lib = device
        .new_library_with_source(source, &options)
        .expect("Compile failed");
    let kernel = lib.get_function("argsort_f32_i32", None).expect("No kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("No pipeline");

    let configs = [256, 512, 1024];

    for n in configs {
        let input_data: Vec<f32> =
            (0..n as usize).map(|i| ((i * 7) % 1000) as f32 / 10.0).collect();
        let input_buf = device.new_buffer(
            (n as usize * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buf = device.new_buffer(
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

        let num_elements = n;
        let top_k = n;
        let sort_order: i32 = 1;

        // Warmup
        for _ in 0..10 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&input_buf), 0);
            enc.set_buffer(1, Some(&output_buf), 0);
            enc.set_bytes(
                2,
                4,
                &num_elements as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                3,
                4,
                &top_k as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                4,
                4,
                &sort_order as *const i32 as *const std::ffi::c_void,
            );
            enc.set_threadgroup_memory_length(0, (n as usize * 4) as u64);
            enc.dispatch_thread_groups(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: n as u64, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let rounds = 500;
        let mut times = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let t = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&input_buf), 0);
            enc.set_buffer(1, Some(&output_buf), 0);
            enc.set_bytes(
                2,
                4,
                &num_elements as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                3,
                4,
                &top_k as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                4,
                4,
                &sort_order as *const i32 as *const std::ffi::c_void,
            );
            enc.set_threadgroup_memory_length(0, (n as usize * 4) as u64);
            enc.dispatch_thread_groups(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: n as u64, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            times.push(t.elapsed().as_nanos() as f64);
        }

        let avg = times.iter().sum::<f64>() / rounds as f64;
        println!("n={:4}: {:8.2} us", n, avg / 1000.0);
    }
    println!();
}

fn bench_conv_2d(device: &Device, queue: &metal::CommandQueue) {
    println!("========================================");
    println!("CONV_2D (Self-written only)");
    println!("========================================");

    let source = include_str!("../../shaders/conv_2d.metal");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(true);
    let lib = device
        .new_library_with_source(source, &options)
        .expect("Compile failed");
    let kernel = lib.get_function("conv_2d_f32_f32", None).expect("No kernel");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("No pipeline");

    let configs = [
        (32, 32, 16, 16, 3, 3, 1, 1),
        (64, 64, 32, 32, 3, 3, 1, 1),
        (64, 128, 32, 32, 3, 3, 1, 1),
        (128, 128, 16, 16, 3, 3, 1, 1),
    ];

    for (in_ch, out_ch, in_h, in_w, ker_h, ker_w, stride, pad) in configs {
        let batch_sz: i32 = 1;
        let out_h = (in_h - ker_h + 2 * pad) / stride + 1;
        let out_w = (in_w - ker_w + 2 * pad) / stride + 1;

        let input_size = (batch_sz * in_ch * in_h * in_w) as usize;
        let weight_size = (out_ch * in_ch * ker_h * ker_w) as usize;
        let output_size = (batch_sz * out_ch * out_h * out_w) as usize;

        let input_data: Vec<f32> = (0..input_size)
            .map(|i| ((i % 1000) as f32 - 500.0) / 100.0)
            .collect();
        let weights: Vec<f32> = (0..weight_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 50.0)
            .collect();

        let input_buf = device.new_buffer(
            (input_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let weight_buf = device.new_buffer(
            (weight_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buf = device.new_buffer(
            (output_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                input_data.as_ptr(),
                input_buf.contents() as *mut f32,
                input_size,
            );
            std::ptr::copy_nonoverlapping(
                weights.as_ptr(),
                weight_buf.contents() as *mut f32,
                weight_size,
            );
        }

        // Warmup
        for _ in 0..5 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&weight_buf), 0);
            enc.set_buffer(1, Some(&input_buf), 0);
            enc.set_buffer(2, Some(&output_buf), 0);
            enc.set_bytes(
                3,
                4,
                &batch_sz as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                4,
                4,
                &in_ch as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                5,
                4,
                &out_ch as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(6, 4, &in_h as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &in_w as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(
                8,
                4,
                &out_h as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                9,
                4,
                &out_w as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                10,
                4,
                &ker_h as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                11,
                4,
                &ker_w as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                12,
                4,
                &stride as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                13,
                4,
                &stride as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                14,
                4,
                &1i32 as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                15,
                4,
                &1i32 as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(16, 4, &pad as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(17, 4, &pad as *const i32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize { width: output_size as u64, height: 1, depth: 1 },
                MTLSize { width: 64, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let rounds = 100;
        let mut times = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let t = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&weight_buf), 0);
            enc.set_buffer(1, Some(&input_buf), 0);
            enc.set_buffer(2, Some(&output_buf), 0);
            enc.set_bytes(
                3,
                4,
                &batch_sz as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                4,
                4,
                &in_ch as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                5,
                4,
                &out_ch as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(6, 4, &in_h as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &in_w as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(
                8,
                4,
                &out_h as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                9,
                4,
                &out_w as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                10,
                4,
                &ker_h as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                11,
                4,
                &ker_w as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                12,
                4,
                &stride as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                13,
                4,
                &stride as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                14,
                4,
                &1i32 as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                15,
                4,
                &1i32 as *const i32 as *const std::ffi::c_void,
            );
            enc.set_bytes(16, 4, &pad as *const i32 as *const std::ffi::c_void);
            enc.set_bytes(17, 4, &pad as *const i32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize { width: output_size as u64, height: 1, depth: 1 },
                MTLSize { width: 64, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            times.push(t.elapsed().as_nanos() as f64);
        }

        let avg = times.iter().sum::<f64>() / rounds as f64;
        let flops = output_size as f64 * (in_ch * ker_h * ker_w) as f64 * 2.0;
        let gflops = flops / (avg * 1e-9) / 1e9;
        println!(
            "{}x{}x{}x{} -> {}x{}x{}: {:8.2} us ({:.1} GFLOPS)",
            batch_sz,
            in_ch,
            in_h,
            in_w,
            out_ch,
            out_h,
            out_w,
            avg / 1000.0,
            gflops
        );
    }
    println!();
}
