// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark: Our F16 GEMV vs llama.cpp NATIVE Metal kernel
// Directly loads llama.cpp's kernel implementation

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
#[repr(C)]
#[derive(Clone, Copy)]
struct GgmlMetalKargsMulMv {
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

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
fn main() {
    use metal::{Device, MTLSize};

    let ctx = MetalContext::new().expect("Failed to create MetalContext");
    let device = ctx.device();

    // Load llama.cpp kernels (short and optimized versions)
    let compile_options = metal::CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    let source = r#"
#include <metal_stdlib>
using namespace metal;

#define N_SIMDWIDTH 32
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

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

// Short version for small K
template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_short_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg) {
    const int r0 = tgpig.x*32 + tiisg;
    const int r1 = tgpig.y;
    if (r0 >= args.ne01) return;

    device const T0 * x = (device const T0 *) (src0 + r0*args.nb01);
    device const T1 * y = (device const T1 *) (src1 + r1*args.nb11);
    device float * dst_f32 = (device float *) dst + (uint64_t)r1*args.ne0;

    float res = 0.0f;
    for (int i = 0; i < args.ne00; ++i) {
        res += (float) x[i] * (float) y[i];
    }
    dst_f32[r0] = res;
}

template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t_short(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_t_t_short_impl<T0, T1, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, tgpig, tiisg);
}

// Optimized version with float4 for large K
template<typename T0, typename T04, typename T1, typename T14, short NR0, typename args_t>
void kernel_mul_mv_t_t_4_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB = 32;
    constexpr short NF = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;
    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;

    device const T14 * y4 = (device const T14 *) (src1 + r1*args.nb11);

    device const T04 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const T04 *) (src0 + (r0 + row)*args.nb01);
    }

    float sumf[NR0] = { [ 0 ... NR0 - 1 ] = 0.0f };

    for (int ib = tiisg; ib < nb; ib += NW) {
        float4 y4b[4] = { y4[ib + 0], y4[ib + 1], y4[ib + 2], y4[ib + 3] };

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            float4 acc = 0.0f;
            for (int j = 0; j < 4; ++j) {
                acc += float4(ax4[row][ib*4 + j]) * y4b[j];
            }
            sumf[row] += acc[0] + acc[1] + acc[2] + acc[3];
        }
    }

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        sumf[row] = simd_sum(sumf[row]);
    }

    if (tiisg == 0) {
        for (short row = 0; row < NR0; ++row) {
            if (r0 + row < args.ne01) {
                ((device float *) dst)[r1*args.ne0 + r0 + row] = sumf[row];
            }
        }
    }
}

template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv_t_t_4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 2, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t_short<half, float>) mul_mv_t_t_short_t;
typedef decltype(kernel_mul_mv_t_t_4<half, half4, float, float4>) mul_mv_t_t_4_t;

template [[host_name("kernel_mul_mv_f16_f32_short")]] kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<half, float>;
template [[host_name("kernel_mul_mv_f16_f32_4")]]    kernel mul_mv_t_t_4_t    kernel_mul_mv_t_t_4<half, half4, float, float4>;
"#;

    let library = device
        .new_library_with_source(source, &compile_options)
        .expect("Failed to compile llama.cpp kernels");

    let short_kernel = library
        .get_function("kernel_mul_mv_f16_f32_short", None)
        .expect("Failed to get short kernel");
    let short_pipeline = device
        .new_compute_pipeline_state_with_function(&short_kernel)
        .expect("Failed to create short pipeline");

    let opt_kernel = library
        .get_function("kernel_mul_mv_f16_f32_4", None)
        .expect("Failed to get optimized kernel");
    let opt_pipeline = device
        .new_compute_pipeline_state_with_function(&opt_kernel)
        .expect("Failed to create optimized pipeline");

    println!(
        "=== F16 GEMV: Our Implementation vs llama.cpp Native Kernel ===\n"
    );

    // Test sizes
    let test_sizes: Vec<(usize, usize)> = vec![
        (128, 4096),
        (512, 4096),
        (1024, 4096),
        (2048, 4096),
        (4096, 4096),
        (8192, 16384),
    ];

    println!("Size (K×M) | Our (μs) | llama.cpp (μs) | Ratio");
    println!("-----------|----------|----------------|-------");

    for (k, m) in test_sizes {
        // Generate test data
        let weights_f16: Vec<u16> = (0..m * k)
            .map(|i| ((i as f32 * 0.1).sin() * 100.0) as u16)
            .collect();
        let input_f32: Vec<f32> =
            (0..k).map(|i| (i as f32 * 0.05).cos()).collect();

        // Create buffers
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
        let output_buffer = device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Benchmark our kernel
        let iterations = 100;

        // Warmup
        for _ in 0..10 {
            let _ = ctx.mv_f16_f32_gpu_weights(
                m,
                k,
                &weights_buffer,
                &input_buffer,
            );
        }

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = ctx.mv_f16_f32_gpu_weights(
                m,
                k,
                &weights_buffer,
                &input_buffer,
            );
        }
        let our_time_us =
            start.elapsed().as_micros() as f64 / iterations as f64;

        // Benchmark llama.cpp kernel
        let args = GgmlMetalKargsMulMv {
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
            nr0: 2,
            r2: 1,
            r3: 1,
        };

        let queue = ctx.queue();

        // Select llama.cpp kernel based on K
        let (pipeline, tg_count) = if k < 1024 {
            (&short_pipeline, (m + 31) / 32)
        } else {
            (&opt_pipeline, (m + 1) / 2)
        };

        // Warmup
        for _ in 0..10 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<GgmlMetalKargsMulMv>() as u64,
                &args as *const GgmlMetalKargsMulMv as *const std::ffi::c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(&output_buffer), 0);
            enc.dispatch_thread_groups(
                MTLSize { width: tg_count as u64, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<GgmlMetalKargsMulMv>() as u64,
                &args as *const GgmlMetalKargsMulMv as *const std::ffi::c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(&output_buffer), 0);
            enc.dispatch_thread_groups(
                MTLSize { width: tg_count as u64, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let llama_time_us =
            start.elapsed().as_micros() as f64 / iterations as f64;

        let ratio = our_time_us / llama_time_us * 100.0;
        println!(
            "{:4}×{:5} | {:8.2} | {:14.2} | {:5.1}%",
            k, m, our_time_us, llama_time_us, ratio
        );
    }

    println!("\n=== Interpretation ===");
    println!("Ratio > 100%: Our kernel is FASTER than llama.cpp");
    println!("Ratio < 100%: Our kernel is SLOWER - need to optimize");
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This benchmark requires Metal support on aarch64");
}
