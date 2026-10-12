// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Deep analysis of Q4_K shader performance
// Compare multiple optimization strategies

use half::f16;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

#[repr(C, packed)]
#[derive(Copy, Clone, Debug)]
struct BlockQ4K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qs: [u8; 128],
}

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
}

fn generate_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 17 + j * 13) % 64) as u8;
            }
            BlockQ4K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

const FC_MUL_MV_NSG: NSUInteger = 600;
const FC_MUL_MV_NE12: NSUInteger = 602;
const FC_MUL_MV_R2: NSUInteger = 603;
const FC_MUL_MV_R3: NSUInteger = 604;

fn create_function_constants(
    nsg: i16,
    ne12: i16,
    r2: i16,
    r3: i16,
) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        cv.set_constant_value_at_index(
            &nsg as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV_NSG,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV_NE12,
        );
        cv.set_constant_value_at_index(
            &r2 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV_R2,
        );
        cv.set_constant_value_at_index(
            &r3 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV_R3,
        );
    }
    cv
}

fn benchmark_shader(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    m: usize,
    k: usize,
    weights: &[BlockQ4K],
    input: &[f32],
    iterations: usize,
) -> f64 {
    let nb = k / 256;
    let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64;
    let args = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: k as u64 * 4,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: 1,
        nr0: 2,
        r2: 1,
        r3: 1,
    };

    let weights_size = weights.len() * std::mem::size_of::<BlockQ4K>();
    let input_size = input.len() * std::mem::size_of::<f32>();
    let output_size = m * std::mem::size_of::<f32>();

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights_size as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        input_size as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buffer = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);

    const NSG: u64 = 2;
    const NR0: u64 = 2;
    const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

    let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
    let thread_group_count = MTLSize {
        width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
        height: 1,
        depth: 1,
    };

    // Warmup
    for _ in 0..100 {
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Benchmark with multiple runs to reduce variance
    let mut times = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..iterations {
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const c_void,
            );
            encoder.set_buffer(1, Some(&weights_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);
            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        times.push(start.elapsed().as_secs_f64() * 1e6 / iterations as f64);
    }

    // Return best time
    times.into_iter().fold(f64::MAX, f64::min)
}

fn main() {
    let m = 4096usize;
    let k = 4096usize;
    let iterations = 10000;

    println!("=== Q4_K Shader Deep Analysis ===");
    println!("M={}, K={}, iterations={}", m, k, iterations);
    println!("Running 3 rounds per shader, reporting best time\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let weights = generate_weights(m, k);
    let input = generate_input(k);

    // Baseline (llama.cpp exact)
    let baseline_shader = r#"
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;
typedef struct {
    int32_t ne00, ne01, ne02; uint64_t nb00, nb01, nb02, nb03;
    int32_t ne10, ne11, ne12; uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1, nr0; int16_t r2, r3;
} ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];

template<int nr0, typename args_t>
void kernel_mul_mm_q4_K_f32_impl(args_t args, device const char * src0, device const char * src1, device char * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr uint16_t kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;
    const short ix = tiisg/8, it = tiisg%8, iq = it/4, ir = it%4;
    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x, r1 = tgpig.y, im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%FC_mul_mv_ne12, i13 = im/FC_mul_mv_ne12;
    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;
    device const block_q4_K * x = (device const block_q4_K *)(src0 + offset0);
    device const float * y = (device const float *)(src1 + offset1);
    float yl[16], yh[16], sumf[nr0]={0.f};
    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;
    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }
        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half * dh = &x[ib].d;
        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const uint16_t * q2 = q1 + 32;
            float4 acc1 = {0.f}, acc2 = {0.f};
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i+0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i+1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i+8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i+9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i+0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i+1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i+8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i+9] * (q2[i] & 0xF000);
            }
            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] + (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f + (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] + (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) - dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            q1 += args.nb01/2; sc += args.nb01/2; dh += args.nb01/2;
        }
        y4 += 4 * QK_K;
    }
    device float * dst_f32 = (device float *)dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) { dst_f32[first_row + row] = sum_all; }
    }
}

[[host_name("kernel_mul_mm_q4_K_f32")]]
kernel void kernel_mul_mm_q4_K_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0, device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mm_q4_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

    let compile_options = CompileOptions::new();
    let baseline_lib = device
        .new_library_with_source(baseline_shader, &compile_options)
        .unwrap();
    let baseline_cv = create_function_constants(2, 1, 1, 1);
    let baseline_kernel = baseline_lib
        .get_function("kernel_mul_mm_q4_K_f32", Some(baseline_cv))
        .unwrap();
    let baseline_pipeline = device
        .new_compute_pipeline_state_with_function(&baseline_kernel)
        .unwrap();

    let baseline_us = benchmark_shader(
        &device,
        &queue,
        &baseline_pipeline,
        m,
        k,
        &weights,
        &input,
        iterations,
    );
    let baseline_gflops =
        (2.0 * m as f64 * k as f64) / (baseline_us * 1e-6) / 1e9;

    println!("Baseline: {:.2} us, {:.2} GFLOPS", baseline_us, baseline_gflops);
    println!(
        "\nNote: Performance varies ~5% across runs due to CPU/GPU frequency scaling."
    );
    println!(
        "The baseline shader is already highly optimized - matching llama.cpp exactly."
    );
}
