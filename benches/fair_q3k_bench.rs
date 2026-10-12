// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Fair benchmark: Self-written vs llama.cpp Q3_K kernel
// Both use the same Rust test harness, same data, same flow
// Run: cargo run --release --bin fair_q3k_bench

use half::f16;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct BlockQ3K {
    hmask: [u8; 32],
    qs: [u8; 64],
    scales: [u8; 12],
    d: u16,
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

fn create_cv() -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();
    unsafe {
        let nsg: i16 = 2;
        let ne12: i16 = 1;
        let r2: i16 = 1;
        let r3: i16 = 1;
        cv.set_constant_value_at_index(
            &nsg as *const i16 as *const c_void,
            MTLDataType::Short,
            600,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const i16 as *const c_void,
            MTLDataType::Short,
            602,
        );
        cv.set_constant_value_at_index(
            &r2 as *const i16 as *const c_void,
            MTLDataType::Short,
            603,
        );
        cv.set_constant_value_at_index(
            &r3 as *const i16 as *const c_void,
            MTLDataType::Short,
            604,
        );
    }
    cv
}

fn main() {
    println!("=== Fair Benchmark: Self-written vs llama.cpp Q3_K ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);

    // llama.cpp Q3_K kernel (exact copy from mul_mv.metal lines 1346-1507)
    let llama_shader = r#"
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define FC_MUL_MV 600
#define N_R0_Q3_K 2

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12  [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2    [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3    [[function_constant(FC_MUL_MV + 4)]];

typedef struct {
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
} ggml_metal_kargs_mul_mv;

typedef struct {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half  d;
} block_q3_K;

template<int nr0, typename args_t>
void kernel_mul_mm_q3_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q3_K * x = (device const block_q3_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float yl[32];

    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short ip  = tid/4;
    const short il  = 2*((tid%4)/2);
    const short ir  = tid%2;
    const short l0  = 8*ir;

    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},
                           {0x0004, 0x0400, 0x0008, 0x0800},
                           {0x0010, 0x1000, 0x0020, 0x2000},
                           {0x0040, 0x4000, 0x0080, 0x8000}};

    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2*ip + il/2];

    const short shift = 2*il;

    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const short q_offset = 32*ip + l0;
    const short y_offset = 128*ip + 32*il + l0;

    device const float * y1 = yy + ix*QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[nr0] = {0.f};
    float sumf2[nr0] = {0.f};

    for (int i = ix; i < nb; i += 4) {
        for (short l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;

        for (short row = 0; row < nr0; ++row) {
            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]);
                s2 += yl[l+1] * (qs & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm[il/2][0]);
                s2 += yl[l+9] * (qs & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            q  += args.nb01/2;
            h  += args.nb01/2;
            a  += args.nb01/2;
            dh += args.nb01/2;
        }

        y1 += 4 * QK_K;
    }

    for (int row = 0; row < nr0; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (tiisg == 0) {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            dst_f32[first_row + row] = sumf1[row];
        }
    }
}

[[host_name("kernel_mul_mm_q3_K_f32")]]
kernel void kernel_mul_mm_q3_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mm_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

    // Self-written template
    let self_shader = include_str!("../src/ops/metal/gemm/mul_mm.metal");

    let llama_lib = device
        .new_library_with_source(llama_shader, &options)
        .expect("llama.cpp shader failed");
    let self_lib = device
        .new_library_with_source(self_shader, &options)
        .expect("self-written shader failed");

    let cv = create_cv();
    let llama_kernel = llama_lib
        .get_function("kernel_mul_mm_q3_K_f32", Some(cv.clone()))
        .expect("llama kernel failed");
    let self_kernel = self_lib
        .get_function("kernel_mul_mm_q3_K_f32", Some(cv.clone()))
        .expect("self kernel failed");

    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_kernel)
        .expect("llama pipeline failed");
    let self_pipeline = device
        .new_compute_pipeline_state_with_function(&self_kernel)
        .expect("self pipeline failed");

    println!("Both pipelines created successfully.\n");

    // Test dimensions
    const M: usize = 4096;
    const K: usize = 4096;
    const NB: usize = K / 256;
    const BLOCK_SIZE: usize = 110;
    const ITERATIONS: usize = 10000;

    // Generate weights (deterministic)
    let mut weights: Vec<BlockQ3K> =
        vec![
            BlockQ3K { hmask: [0; 32], qs: [0; 64], scales: [0; 12], d: 0 };
            M * NB
        ];
    for i in 0..(M * NB) {
        for j in 0..32 {
            weights[i].hmask[j] = ((i * 17 + j * 23) % 256) as u8;
        }
        for j in 0..64 {
            weights[i].qs[j] = ((i * 13 + j * 7) % 256) as u8;
        }
        for j in 0..12 {
            weights[i].scales[j] = ((i * 31 + j * 11) % 128) as u8;
        }
        weights[i].d = f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits();
    }

    // Generate input (deterministic)
    let input: Vec<f32> = (0..K).map(|i| (i % 100) as f32 * 0.01).collect();

    // Create buffers
    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (M * NB * BLOCK_SIZE) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (K * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let llama_output = device
        .new_buffer((M * 4) as u64, MTLResourceOptions::StorageModeShared);
    let self_output = device
        .new_buffer((M * 4) as u64, MTLResourceOptions::StorageModeShared);

    // Args
    let args = MvArgs {
        ne00: K as i32,
        ne01: M as i32,
        ne02: 1,
        nb00: BLOCK_SIZE as u64,
        nb01: (NB * BLOCK_SIZE) as u64,
        nb02: 0,
        nb03: 0,
        ne10: K as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: (K * 4) as u64,
        nb12: 0,
        nb13: 0,
        ne0: M as i32,
        ne1: 1,
        nr0: 2,
        r2: 1,
        r3: 1,
    };
    let args_buf = device.new_buffer_with_data(
        &args as *const MvArgs as *const c_void,
        std::mem::size_of::<MvArgs>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Thread config
    const NSG: u64 = 2;
    const NR0: u64 = 2;
    let tg_count = MTLSize::new((M as u64 + NSG * NR0 - 1) / (NSG * NR0), 1, 1);
    let tg_size = MTLSize::new(32, NSG, 1);

    // Warmup both
    print!("Warming up... ");
    for _ in 0..100 {
        for (pipeline, output) in [&llama_pipeline, &self_pipeline]
            .iter()
            .zip([&llama_output, &self_output])
        {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(&args_buf), 0);
            enc.set_buffer(1, Some(&weights_buf), 0);
            enc.set_buffer(2, Some(&input_buf), 0);
            enc.set_buffer(3, Some(output), 0);
            enc.dispatch_thread_groups(tg_count, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }
    println!("done");

    // Benchmark llama.cpp
    println!("\nBenchmarking llama.cpp ({} iterations)...", ITERATIONS);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_buffer(0, Some(&args_buf), 0);
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&llama_output), 0);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let llama_time = start.elapsed().as_micros() as f64;
    let llama_avg = llama_time / ITERATIONS as f64;
    let llama_gflops = 2.0 * M as f64 * K as f64 / (llama_avg * 1e-6) / 1e9;

    // Benchmark self-written
    println!("Benchmarking self-written ({} iterations)...", ITERATIONS);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self_pipeline);
        enc.set_buffer(0, Some(&args_buf), 0);
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&self_output), 0);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let self_time = start.elapsed().as_micros() as f64;
    let self_avg = self_time / ITERATIONS as f64;
    let self_gflops = 2.0 * M as f64 * K as f64 / (self_avg * 1e-6) / 1e9;

    // Check correctness
    let llama_ptr = unsafe {
        std::slice::from_raw_parts(llama_output.contents() as *const f32, M)
    };
    let self_ptr = unsafe {
        std::slice::from_raw_parts(self_output.contents() as *const f32, M)
    };
    let correct = (0..M)
        .filter(|&i| {
            let lv = llama_ptr[i];
            let sv = self_ptr[i];
            if lv.abs() > 1e-6 {
                (lv - sv).abs() / lv.abs() < 0.01
            } else {
                (lv - sv).abs() < 1e-6
            }
        })
        .count();

    println!("\n=== Results ===");
    println!("{:15} {:>12} {:>10}", "", "llama.cpp", "self");
    println!("{:15} {:>12.2} us {:>10.2} us", "Avg time", llama_avg, self_avg);
    println!("{:15} {:>12.1}    {:>10.1}", "GFLOPS", llama_gflops, self_gflops);
    println!(
        "{:15} {:>12.1}%   {:>10}",
        "Ratio",
        llama_avg / self_avg * 100.0,
        ""
    );
    println!(
        "\nCorrectness: {}/{} ({:.1}%)",
        correct,
        M,
        correct as f32 / M as f32 * 100.0
    );

    if correct == M {
        println!("✓ 100% correct");
    }
}
