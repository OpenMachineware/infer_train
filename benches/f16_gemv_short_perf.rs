// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Performance comparison: Our F16 GEMV short kernel vs reference
// Tests K < 32 range, alternating A-B-A-B pattern

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use metal::{Device, MTLSize};
    use std::time::Instant;

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let compile_options = metal::CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    // Our implementation
    let our_source =
        include_str!("../src/ops/metal/gemv/mul_mv_f16_kernels.metal");
    let our_library = device
        .new_library_with_source(our_source, &compile_options)
        .expect("Failed to compile our library");
    let our_kernel = our_library
        .get_function("kernel_mul_mv_f16_f32_short", None)
        .expect("Failed to get our kernel");
    let our_pipeline = device
        .new_compute_pipeline_state_with_function(&our_kernel)
        .expect("Failed to create our pipeline");

    // Reference implementation
    let ref_source = r#"
#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32

struct ggml_metal_kargs_mul_mv {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
};

kernel void kernel_mul_mv_f16_f32_short(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int r0 = tgpig.x*32 + tiisg;
    const int r1 = tgpig.y;
    if (r0 >= args.ne01) return;

    device const half * x = (device const half *) (src0 + r0*args.nb01);
    device const float * y = (device const float *) (src1 + r1*args.nb11);
    device float * dst_f32 = (device float *) dst + (uint64_t)r1*args.ne0;

    float res = 0.0f;
    for (int i = 0; i < args.ne00; ++i) {
        res += (float) x[i] * (float) y[i];
    }
    dst_f32[r0] = res;
}
"#;
    let ref_library = device
        .new_library_with_source(ref_source, &compile_options)
        .expect("Failed to compile reference library");
    let ref_kernel = ref_library
        .get_function("kernel_mul_mv_f16_f32_short", None)
        .expect("Failed to get reference kernel");
    let ref_pipeline = device
        .new_compute_pipeline_state_with_function(&ref_kernel)
        .expect("Failed to create reference pipeline");

    #[repr(C)]
    struct MvArgs {
        ne00: i32,
        ne01: i32,
        ne02: i32,
        nb00: u64,
        nb01: u64,
        nb02: u64,
        nb03: u64,
        ne10: i32,
        ne11: i32,
        ne12: i32,
        nb10: u64,
        nb11: u64,
        nb12: u64,
        nb13: u64,
        ne0: i32,
        ne1: i32,
        nr0: i32,
        r2: i16,
        r3: i16,
        nsg: i16,
        ne12_batch: i16,
    }

    #[repr(C)]
    struct LlamaArgs {
        ne00: i32,
        ne01: i32,
        ne02: i32,
        nb00: u64,
        nb01: u64,
        nb02: u64,
        nb03: u64,
        ne10: i32,
        ne11: i32,
        ne12: i32,
        nb10: u64,
        nb11: u64,
        nb12: u64,
        nb13: u64,
        ne0: i32,
        ne1: i32,
        nr0: i32,
        r2: i16,
        r3: i16,
    }

    println!("F16 GEMV Short Kernel Performance Test (K < 32)");
    println!("==================================================\n");

    // Test sizes: K values < 32, M = 64
    let test_cases = [(16, 64), (24, 64), (31, 64)];
    let warmup_iters = 50;
    let test_iters = 500;

    for (k, m) in test_cases {
        println!("K={}, M={}", k, m);

        // Generate test data
        let weights_f16: Vec<u16> = (0..m * k)
            .map(|i| {
                half::f16::from_f32((i % 100) as f32 * 0.1 - 5.0).to_bits()
            })
            .collect();
        let input_f32: Vec<f32> =
            (0..k).map(|i| (i % 50) as f32 * 0.05 - 1.25).collect();

        let weights_buffer = device.new_buffer_with_data(
            weights_f16.as_ptr() as *const std::ffi::c_void,
            (m * k * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = device.new_buffer_with_data(
            input_f32.as_ptr() as *const std::ffi::c_void,
            (k * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_our = device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_ref = device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 2,
            nb01: (k * 2) as u64,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: (k * 4) as u64,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 32,
            r2: 1,
            r3: 1,
            nsg: 1,
            ne12_batch: 1,
        };

        let llama_args = LlamaArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 2,
            nb01: (k * 2) as u64,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: (k * 4) as u64,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 32,
            r2: 1,
            r3: 1,
        };

        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: ((m + 31) / 32) as u64, height: 1, depth: 1 };

        // Warmup - alternate A-B
        for _ in 0..warmup_iters {
            // Our implementation
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&our_pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const std::ffi::c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(&output_our), 0);
            enc.dispatch_thread_groups(thread_group_count, thread_group_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Reference implementation
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ref_pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<LlamaArgs>() as u64,
                &llama_args as *const LlamaArgs as *const std::ffi::c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(&output_ref), 0);
            enc.dispatch_thread_groups(thread_group_count, thread_group_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Performance test - alternate A-B-A-B
        let mut our_times = Vec::new();
        let mut ref_times = Vec::new();

        for _ in 0..test_iters {
            // Our implementation
            let start = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&our_pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const std::ffi::c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(&output_our), 0);
            enc.dispatch_thread_groups(thread_group_count, thread_group_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            our_times.push(start.elapsed());

            // Reference implementation
            let start = Instant::now();
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ref_pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<LlamaArgs>() as u64,
                &llama_args as *const LlamaArgs as *const std::ffi::c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(&output_ref), 0);
            enc.dispatch_thread_groups(thread_group_count, thread_group_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            ref_times.push(start.elapsed());
        }

        // Calculate average times
        let our_avg: std::time::Duration =
            our_times.iter().sum::<std::time::Duration>() / test_iters as u32;
        let ref_avg: std::time::Duration =
            ref_times.iter().sum::<std::time::Duration>() / test_iters as u32;

        // Calculate GB/s and GFLOPS
        let bytes = (m * k * 2 + k * 4 + m * 4) as f64; // weights + input + output
        let flops = (m * k * 2) as f64; // multiply + add

        let our_gbs = bytes / our_avg.as_secs_f64() / 1e9;
        let ref_gbs = bytes / ref_avg.as_secs_f64() / 1e9;
        let our_gflops = flops / our_avg.as_secs_f64() / 1e9;
        let ref_gflops = flops / ref_avg.as_secs_f64() / 1e9;

        let speedup = ref_avg.as_secs_f64() / our_avg.as_secs_f64();

        println!(
            "  Our:  {:?} ({:.1} GB/s, {:.1} GFLOPS)",
            our_avg, our_gbs, our_gflops
        );
        println!(
            "  Ref:  {:?} ({:.1} GB/s, {:.1} GFLOPS)",
            ref_avg, ref_gbs, ref_gflops
        );
        println!("  Speedup: {:.2}x", speedup);

        if speedup < 1.0 {
            println!("  ⚠️  {:.1}% slower", (1.0 - speedup) * 100.0);
        } else {
            println!("  ✓ {:.1}% faster", (speedup - 1.0) * 100.0);
        }
        println!();
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This test requires Metal support on aarch64");
}
