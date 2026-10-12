// Fair Q6_K benchmark: llama.cpp vs self-written
// Same environment, same data, same function constants, same threadgroups

use half::f16;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

const FC_MUL_MV: NSUInteger = 600;
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

#[repr(C)]
#[derive(Copy, Clone)]
struct BlockQ6K {
    ql: [u8; 128],
    qh: [u8; 64],
    scales: [i8; 16],
    d: u16, // half stored as u16
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

    // llama.cpp Q6_K kernel (exact code from mul_mv.metal)
    let llama_shader = r#"
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define N_R0_Q6_K 2

constant short FC_mul_mv_nsg   [[function_constant(600 + 0)]];
constant short FC_mul_mv_ne12  [[function_constant(600 + 2)]];
constant short FC_mul_mv_r2    [[function_constant(600 + 3)]];
constant short FC_mul_mv_r3    [[function_constant(600 + 4)]];

typedef struct {
    int32_t  ne00; int32_t  ne01; int32_t  ne02;
    uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
    int32_t  ne10; int32_t  ne11; int32_t  ne12;
    uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
    int32_t  ne0;  int32_t  ne1;  int32_t  nr0;
    int16_t  r2;   int16_t  r3;
} ggml_metal_kargs_mul_mv;

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half  d;
} block_q6_K;

template<int nr0, typename args_t>
void kernel_gemm_q6_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0] = { 0.f };

    float yl[16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;
        device const half    * dh = &x[i].d;

        device const float * y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01/2;
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_gemm_q6_K_f32")]]
kernel void kernel_gemm_q6_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_gemm_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

    // Self-written Q6_K kernel (from gemm.metal)
    let my_shader = r#"
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant short FC_nsg   [[function_constant(600)]];
constant short FC_ne12  [[function_constant(602)]];
constant short FC_r2    [[function_constant(603)]];
constant short FC_r3    [[function_constant(604)]];

typedef struct {
    int32_t  ne00; int32_t  ne01; int32_t  ne02;
    uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
    int32_t  ne10; int32_t  ne11; int32_t  ne12;
    uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
    int32_t  ne0;  int32_t  ne1;  int32_t  nr0;
    int16_t  r2;   int16_t  r3;
} ggml_metal_kargs_mul_mv;

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half  d;
} block_q6_K;

template<int nr0, typename args_t>
void kernel_gemm_q6_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    const short NSG = FC_nsg;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00 / QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im % FC_ne12;
    const uint i13 = im / FC_ne12;

    const uint64_t offset0 = first_row * args.nb01 + (i12 / FC_r2) * args.nb02 + (i13 / FC_r3) * args.nb03;
    const uint64_t offset1 = r1 * args.nb11 + i12 * args.nb12 + i13 * args.nb13;

    device const block_q6_K * x = (device const block_q6_K *)(src0 + offset0);
    device const float      * yy = (device const float      *)(src1 + offset1);

    float sumf[nr0] = {0.f};

    float yl[16];

    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;

    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;
        device const half    * dh = &x[i].d;

        device const float * y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
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

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01 / 2;
        }
    }

    device float * dst_f32 = (device float *)dst + (uint64_t)im * args.ne0 * args.ne1 + (uint64_t)r1 * args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_gemm_q6_K_f32")]]
kernel void kernel_gemm_q6_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_gemm_q6_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

    let compile_options = CompileOptions::new();

    let llama_lib = device
        .new_library_with_source(llama_shader, &compile_options)
        .expect("llama.cpp compilation failed");

    let my_lib = device
        .new_library_with_source(my_shader, &compile_options)
        .expect("Self-written compilation failed");

    // Use correct function constants: NSG=2, NE12=1, R2=1, R3=1
    let cv = create_function_constants(2, 1, 1, 1);

    let llama_func = llama_lib
        .get_function("kernel_gemm_q6_K_f32", Some(cv.clone()))
        .unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_func).unwrap();

    let my_func =
        my_lib.get_function("kernel_gemm_q6_K_f32", Some(cv)).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_func).unwrap();

    println!("Pipelines created with NSG=2 (llama.cpp default for Q6_K).\n");

    // Test dimensions
    let m = 1024;
    let k = 4096;
    let n = 1;
    let nb = k / 256;

    // Generate Q6_K weights
    let weights: Vec<BlockQ6K> = (0..m * nb)
        .map(|i| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0i8; 16];
            for j in 0..128 {
                ql[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            for j in 0..64 {
                qh[j] = ((i * 19 + j * 13) % 256) as u8;
            }
            for j in 0..16 {
                scales[j] = ((i * 17 + j * 11) % 64) as i8 - 32;
            }
            BlockQ6K {
                ql,
                qh,
                scales,
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            }
        })
        .collect();

    let input: Vec<f32> = (0..k).map(|i| ((i % 100) as f32) * 0.01).collect();

    // Create buffers
    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const _,
        (m * nb * std::mem::size_of::<BlockQ6K>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const _,
        (k * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let llama_output_buf = device.new_buffer(
        (m * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let my_output_buf = device.new_buffer(
        (m * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let args = MulMVArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01: (nb * std::mem::size_of::<BlockQ6K>()) as u64,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: n as i32,
        ne12: 1,
        nb10: 4,
        nb11: (k * 4) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: n as i32,
        nr0: 1,
        r2: 1,
        r3: 1,
    };

    // Threadgroup configuration: NSG=2, N_R0=2
    // Grid: (ceil(m / (NSG * N_R0)), n, 1) = (ceil(1024 / 4), 1, 1) = (256, 1, 1)
    // Threadgroup size: NSG * 32 = 64
    let grid = MTLSize::new(256, 1, 1);
    let tg = MTLSize::new(64, 1, 1);

    // Warmup
    for _ in 0..10 {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&llama_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&input_buf), 0);
        encoder.set_buffer(3, Some(&llama_output_buf), 0);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    // Alternating benchmark to minimize thermal effects
    const ITERATIONS: usize = 500;
    let mut llama_times = Vec::with_capacity(ITERATIONS);
    let mut my_times = Vec::with_capacity(ITERATIONS);

    for _ in 0..ITERATIONS {
        // llama.cpp
        let start = Instant::now();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&llama_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&input_buf), 0);
        encoder.set_buffer(3, Some(&llama_output_buf), 0);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        llama_times.push(start.elapsed().as_nanos() as f64);

        // Self-written
        let start = Instant::now();
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&my_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMVArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buf), 0);
        encoder.set_buffer(2, Some(&input_buf), 0);
        encoder.set_buffer(3, Some(&my_output_buf), 0);
        encoder.dispatch_thread_groups(grid, tg);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        my_times.push(start.elapsed().as_nanos() as f64);
    }

    // Calculate results
    llama_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    my_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let llama_median = llama_times[ITERATIONS / 2];
    let my_median = my_times[ITERATIONS / 2];
    let ratio = my_median / llama_median * 100.0;

    println!("Q6_K Fair Benchmark (M={}, K={}, NSG=2)", m, k);
    println!("==========================================");
    println!("llama.cpp:  {:>8.2} us (median)", llama_median / 1000.0);
    println!("Self-written: {:>8.2} us (median)", my_median / 1000.0);
    println!("Ratio:       {:>8.1}%", ratio);

    // Verify correctness
    let llama_output = unsafe {
        std::slice::from_raw_parts(llama_output_buf.contents() as *const f32, m)
    };
    let my_output = unsafe {
        std::slice::from_raw_parts(my_output_buf.contents() as *const f32, m)
    };

    let mut correct = 0;
    let mut max_rel_err = 0.0f64;
    for i in 0..m {
        let rel_err = if llama_output[i].abs() > 1e-6 {
            (my_output[i] - llama_output[i]).abs() / llama_output[i].abs()
        } else {
            (my_output[i] - llama_output[i]).abs()
        };
        if rel_err < 0.01 {
            correct += 1;
        }
        max_rel_err = max_rel_err.max(rel_err as f64);
    }

    println!(
        "\nCorrectness: {}/{} ({:.1}%), max relative error: {:.6}",
        correct,
        m,
        correct as f64 / m as f64 * 100.0,
        max_rel_err
    );
}
