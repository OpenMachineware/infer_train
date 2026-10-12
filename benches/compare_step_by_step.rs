// Direct comparison: each step timing
// llama.cpp vs self-written, same data, same environment

use metal::*;
use std::ffi::c_void;
use std::time::Instant;

const FC_MUL_MV: NSUInteger = 600;

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
            FC_MUL_MV + 0,
        );
        cv.set_constant_value_at_index(
            &ne12 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV + 2,
        );
        cv.set_constant_value_at_index(
            &r2 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV + 3,
        );
        cv.set_constant_value_at_index(
            &r3 as *const i16 as *const c_void,
            MTLDataType::Short,
            FC_MUL_MV + 4,
        );
    }
    cv
}

#[repr(C)]
struct MulMVArgs {
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

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Q6_K kernel code (EXACT from llama.cpp mul_mv.metal lines 1762-1868)
    let llama_shader = r#"
#include <metal_stdlib>
using namespace metal;
#define QK_K 256
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define N_R0_Q6_K 2

constant short FC_mul_mv_nsg [[function_constant(600 + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(600 + 2)]];
constant short FC_mul_mv_r2 [[function_constant(600 + 3)]];
constant short FC_mul_mv_r3 [[function_constant(600 + 4)]];

typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; } block_q6_K;

template<int nr0, typename args_t>
void kernel_gemm_q6_K_f32_impl(args_t args, device const char * src0, device const char * src1, device char * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr uint8_t kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x, r1 = tgpig.y, im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%FC_mul_mv_ne12, i13 = im/FC_mul_mv_ne12;
    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;
    device const block_q6_K * x = (device const block_q6_K *)(src0 + offset0);
    device const float * yy = (device const float *)(src1 + offset1);
    float sumf[nr0] = {0.f};
    float yl[16];
    const short tid = tiisg/2, ix = tiisg%2, ip = tid/8, il = tid%8, l0 = 4*il, is = 8*ip + l0/16;
    const short y_offset = 128*ip + l0, q_offset_l = 64*ip + l0, q_offset_h = 32*ip + l0;
    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t * sc = x[i].scales + is;
        device const half * dh = &x[i].d;
        device const float * y = yy + i * QK_K + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l + 0]; yl[4*l + 1] = y[l + 32]; yl[4*l + 2] = y[l + 64]; yl[4*l + 3] = y[l + 96];
        }
        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
            q1 += args.nb01; q2 += args.nb01; qh += args.nb01; sc += args.nb01; dh += args.nb01/2;
        }
    }
    device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) { dst_f32[first_row + row] = sum_all; }
    }
}
[[host_name("kernel_gemm_q6_K_f32")]]
kernel void kernel_gemm_q6_K_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0, device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_gemm_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

    // Self-written Q6_K kernel (from gemm.metal)
    let my_shader = r#"
#include <metal_stdlib>
using namespace metal;
#define QK_K 256
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant short FC_nsg [[function_constant(600)]];
constant short FC_ne12 [[function_constant(602)]];
constant short FC_r2 [[function_constant(603)]];
constant short FC_r3 [[function_constant(604)]];

typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; } block_q6_K;

template<int nr0, typename args_t>
void kernel_gemm_q6_K_f32_impl(args_t args, device const char * src0, device const char * src1, device char * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_nsg;
    constexpr uint8_t kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
    const int nb = args.ne00 / QK_K;
    const int r0 = tgpig.x, r1 = tgpig.y, im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im % FC_ne12, i13 = im / FC_ne12;
    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;
    device const block_q6_K * x = (device const block_q6_K *)(src0 + offset0);
    device const float * yy = (device const float *)(src1 + offset1);
    float sumf[nr0] = {0.f};
    float yl[16];
    const short tid = tiisg / 2, ix = tiisg % 2, ip = tid / 8, il = tid % 8, l0 = 4 * il, is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0, q_offset_l = 64 * ip + l0, q_offset_h = 32 * ip + l0;
    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t * sc = x[i].scales + is;
        device const half * dh = &x[i].d;
        device const float * y = yy + i * QK_K + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l + 0]; yl[4*l + 1] = y[l + 32]; yl[4*l + 2] = y[l + 64]; yl[4*l + 3] = y[l + 96];
        }
        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l] >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l] >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
            q1 += args.nb01; q2 += args.nb01; qh += args.nb01; sc += args.nb01; dh += args.nb01 / 2;
        }
    }
    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) { dst_f32[first_row + row] = sum_all; }
    }
}
[[host_name("kernel_gemm_q6_K_f32")]]
kernel void kernel_gemm_q6_K_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0, device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_gemm_q6_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

    // Compile both
    let compile_options = CompileOptions::new();
    let llama_lib = device
        .new_library_with_source(llama_shader, &compile_options)
        .expect("llama.cpp compile failed");
    let my_lib = device
        .new_library_with_source(my_shader, &compile_options)
        .expect("Self-written compile failed");

    let cv = create_function_constants(2, 1, 1, 1);
    let llama_func = llama_lib
        .get_function("kernel_gemm_q6_K_f32", Some(cv.clone()))
        .unwrap();
    let my_func =
        my_lib.get_function("kernel_gemm_q6_K_f32", Some(cv)).unwrap();

    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_func).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_func).unwrap();

    let m = 1024;
    let k = 4096;
    let nb = k / 256;
    let block_size = 210;

    // Shared data
    let weights_size = m * nb * block_size;
    let mut weights_data = vec![0u8; weights_size];
    for i in 0..weights_size {
        weights_data[i] = ((i * 17 + 13) % 256) as u8;
    }
    let input: Vec<f32> = (0..k).map(|i| ((i % 100) as f32) * 0.01).collect();

    let args = MulMVArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01: (nb * block_size) as u64,
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
        nr0: 1,
        r2: 1,
        r3: 1,
    };

    let grid = MTLSize::new(256, 1, 1);
    let tg = MTLSize::new(64, 1, 1);

    println!("=== Step-by-Step Direct Comparison ===\n");
    println!(
        "{:20} | {:>12} | {:>12} | {:>8}",
        "Step", "llama.cpp", "Self-written", "Ratio"
    );

    // STEP 1: Compile time (library compilation)
    let t1_llama = Instant::now();
    let _llama_lib2 =
        device.new_library_with_source(llama_shader, &compile_options).unwrap();
    let llama_compile_time = t1_llama.elapsed();

    let t1_my = Instant::now();
    let _my_lib2 =
        device.new_library_with_source(my_shader, &compile_options).unwrap();
    let my_compile_time = t1_my.elapsed();

    println!(
        "{:20} | {:>10.2} ms | {:>10.2} ms | {:>7.1}%",
        "Shader compile",
        llama_compile_time.as_secs_f64() * 1000.0,
        my_compile_time.as_secs_f64() * 1000.0,
        my_compile_time.as_secs_f64() / llama_compile_time.as_secs_f64()
            * 100.0
    );

    // STEP 2: Pipeline creation
    let t2_llama = Instant::now();
    let _llama_pipe2 =
        device.new_compute_pipeline_state_with_function(&llama_func).unwrap();
    let llama_pipeline_time = t2_llama.elapsed();

    let t2_my = Instant::now();
    let _my_pipe2 =
        device.new_compute_pipeline_state_with_function(&my_func).unwrap();
    let my_pipeline_time = t2_my.elapsed();

    println!(
        "{:20} | {:>10.2} ms | {:>10.2} ms | {:>7.1}%",
        "Pipeline create",
        llama_pipeline_time.as_secs_f64() * 1000.0,
        my_pipeline_time.as_secs_f64() * 1000.0,
        my_pipeline_time.as_secs_f64() / llama_pipeline_time.as_secs_f64()
            * 100.0
    );

    // STEP 3: Buffer allocation
    let t3_llama = Instant::now();
    let llama_weights_buf = device
        .new_buffer(weights_size as u64, MTLResourceOptions::StorageModeShared);
    let llama_input_buf = device
        .new_buffer((k * 4) as u64, MTLResourceOptions::StorageModeShared);
    let llama_output_buf = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);
    let llama_buf_alloc_time = t3_llama.elapsed();

    let t3_my = Instant::now();
    let my_weights_buf = device
        .new_buffer(weights_size as u64, MTLResourceOptions::StorageModeShared);
    let my_input_buf = device
        .new_buffer((k * 4) as u64, MTLResourceOptions::StorageModeShared);
    let my_output_buf = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);
    let my_buf_alloc_time = t3_my.elapsed();

    println!(
        "{:20} | {:>10.2} µs | {:>10.2} µs | {:>7.1}%",
        "Buffer alloc",
        llama_buf_alloc_time.as_secs_f64() * 1e6,
        my_buf_alloc_time.as_secs_f64() * 1e6,
        my_buf_alloc_time.as_secs_f64() / llama_buf_alloc_time.as_secs_f64()
            * 100.0
    );

    // STEP 4: CPU → GPU write
    let t4_llama = Instant::now();
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights_data.as_ptr(),
            llama_weights_buf.contents() as *mut u8,
            weights_size,
        );
        std::ptr::copy_nonoverlapping(
            input.as_ptr(),
            llama_input_buf.contents() as *mut f32,
            k,
        );
    }
    let llama_write_time = t4_llama.elapsed();

    let t4_my = Instant::now();
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights_data.as_ptr(),
            my_weights_buf.contents() as *mut u8,
            weights_size,
        );
        std::ptr::copy_nonoverlapping(
            input.as_ptr(),
            my_input_buf.contents() as *mut f32,
            k,
        );
    }
    let my_write_time = t4_my.elapsed();

    println!(
        "{:20} | {:>10.2} µs | {:>10.2} µs | {:>7.1}%",
        "CPU→GPU write",
        llama_write_time.as_secs_f64() * 1e6,
        my_write_time.as_secs_f64() * 1e6,
        my_write_time.as_secs_f64() / llama_write_time.as_secs_f64() * 100.0
    );

    // Warmup both
    for _ in 0..10 {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&llama_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&llama_weights_buf), 0);
        encoder.set_buffer(2, Some(&llama_input_buf), 0);
        encoder.set_buffer(3, Some(&llama_output_buf), 0);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    for _ in 0..10 {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&my_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&my_weights_buf), 0);
        encoder.set_buffer(2, Some(&my_input_buf), 0);
        encoder.set_buffer(3, Some(&my_output_buf), 0);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    // STEP 5: GPU kernel execution (alternating to minimize thermal effects)
    const ITERATIONS: usize = 200;
    let mut llama_gpu_times = Vec::with_capacity(ITERATIONS);
    let mut my_gpu_times = Vec::with_capacity(ITERATIONS);

    for i in 0..ITERATIONS {
        // llama.cpp
        let t_llama = Instant::now();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&llama_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&llama_weights_buf), 0);
        encoder.set_buffer(2, Some(&llama_input_buf), 0);
        encoder.set_buffer(3, Some(&llama_output_buf), 0);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        llama_gpu_times.push(t_llama.elapsed().as_nanos() as f64);

        // Self-written
        let t_my = Instant::now();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&my_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&my_weights_buf), 0);
        encoder.set_buffer(2, Some(&my_input_buf), 0);
        encoder.set_buffer(3, Some(&my_output_buf), 0);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        my_gpu_times.push(t_my.elapsed().as_nanos() as f64);
    }

    llama_gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    my_gpu_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let llama_median_gpu = llama_gpu_times[ITERATIONS / 2];
    let my_median_gpu = my_gpu_times[ITERATIONS / 2];

    println!(
        "{:20} | {:>10.2} µs | {:>10.2} µs | {:>7.1}%",
        "GPU kernel exec",
        llama_median_gpu / 1000.0,
        my_median_gpu / 1000.0,
        my_median_gpu / llama_median_gpu * 100.0
    );

    // STEP 6: GPU → CPU read
    let t6_llama = Instant::now();
    let llama_output: &[f32] = unsafe {
        std::slice::from_raw_parts(llama_output_buf.contents() as *const f32, m)
    };
    let _llama_sum: f64 = llama_output.iter().map(|&x| x as f64).sum();
    let llama_read_time = t6_llama.elapsed();

    let t6_my = Instant::now();
    let my_output: &[f32] = unsafe {
        std::slice::from_raw_parts(my_output_buf.contents() as *const f32, m)
    };
    let _my_sum: f64 = my_output.iter().map(|&x| x as f64).sum();
    let my_read_time = t6_my.elapsed();

    println!(
        "{:20} | {:>10.2} ns | {:>10.2} ns | {:>7.1}%",
        "GPU→CPU read",
        llama_read_time.as_secs_f64() * 1e9,
        my_read_time.as_secs_f64() * 1e9,
        my_read_time.as_secs_f64() / llama_read_time.as_secs_f64() * 100.0
    );

    println!("\n=== Summary ===");
    if my_median_gpu > llama_median_gpu {
        println!(
            "Self-written GPU kernel is {:.1}% SLOWER than llama.cpp",
            (my_median_gpu / llama_median_gpu - 1.0) * 100.0
        );
    } else {
        println!(
            "Self-written GPU kernel is {:.1}% FASTER than llama.cpp",
            (1.0 - my_median_gpu / llama_median_gpu) * 100.0
        );
    }
}
