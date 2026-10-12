use half::f16;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct BlockQ6K {
    ql: [u8; 128],
    qh: [u8; 64],
    scales: [i8; 16],
    d: u16,
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

fn main() {
    println!("=== Q6_K MV: llama.cpp vs self-written ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // llama.cpp Q6_K kernel (exact copy from mul_mv.metal lines 1762-1867)
    let llama_shader = r#"
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define N_R0_Q6_K 2

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
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    half    d;
} block_q6_K;

template<int nr0, typename args_t>
void kernel_mul_mm_q6_K_f32_impl(
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

[[host_name("kernel_mul_mm_q6_K_f32")]]
kernel void kernel_mul_mm_q6_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mm_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

    let compile_options = CompileOptions::new();
    let llama_lib =
        match device.new_library_with_source(llama_shader, &compile_options) {
            Ok(lib) => {
                println!("llama.cpp shader compiled");
                lib
            }
            Err(e) => {
                println!("llama.cpp compilation failed: {:?}", e);
                return;
            }
        };

    // Load self-written shader
    let my_source = include_str!("../src/ops/metal/gemm/mul_mm.metal");
    let my_lib =
        match device.new_library_with_source(my_source, &compile_options) {
            Ok(lib) => {
                println!("Self-written shader compiled");
                lib
            }
            Err(e) => {
                println!("Self-written compilation failed: {:?}", e);
                return;
            }
        };

    // Create pipelines
    let cv = create_function_constants(1, 1, 1, 1);

    let llama_func = llama_lib
        .get_function("kernel_mul_mm_q6_K_f32", Some(cv.clone()))
        .unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_func).unwrap();

    let my_func = my_lib
        .get_function("kernel_mul_mm_q6_K_f32", Some(cv.clone()))
        .unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_func).unwrap();

    println!("Pipelines created.\n");

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
                scales[j] = ((i * 17 + j * 11) % 64) as i8;
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

    let args_buf = device.new_buffer_with_data(
        &args as *const _ as *const _,
        std::mem::size_of::<MulMVArgs>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Use same threadgroup count as before (verified 100% correct)
    let thread_groups = MTLSize::new(m as u64, n as u64, 1);
    let thread_group_size = MTLSize::new(32, 1, 1);

    // Run llama.cpp kernel
    let cmd_buffer = queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&llama_pipeline);
    encoder.set_buffer(0, Some(&args_buf), 0);
    encoder.set_buffer(1, Some(&weights_buf), 0);
    encoder.set_buffer(2, Some(&input_buf), 0);
    encoder.set_buffer(3, Some(&llama_output_buf), 0);
    encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    // Run self-written kernel
    let cmd_buffer = queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&my_pipeline);
    encoder.set_buffer(0, Some(&args_buf), 0);
    encoder.set_buffer(1, Some(&weights_buf), 0);
    encoder.set_buffer(2, Some(&input_buf), 0);
    encoder.set_buffer(3, Some(&my_output_buf), 0);
    encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    // Compare results
    let llama_output = unsafe {
        std::slice::from_raw_parts(llama_output_buf.contents() as *const f32, m)
    };
    let my_output = unsafe {
        std::slice::from_raw_parts(my_output_buf.contents() as *const f32, m)
    };

    let mut correct_count = 0;
    let mut max_rel_err = 0.0f32;
    for i in 0..m {
        let rel_err = if llama_output[i] != 0.0 {
            (my_output[i] - llama_output[i]).abs() / llama_output[i].abs()
        } else {
            my_output[i].abs()
        };
        if rel_err < 0.01 {
            correct_count += 1;
        }
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }
    }

    println!(
        "Correctness: {}/{} ({:.1}%)",
        correct_count,
        m,
        correct_count as f32 / m as f32 * 100.0
    );
    println!("Max relative error: {:.6}", max_rel_err);

    if correct_count == m {
        println!("✓ 100% correct! Running performance test...\n");

        let iterations = 1000;

        // Benchmark llama.cpp
        let start = Instant::now();
        for _ in 0..iterations {
            let cmd_buffer = queue.new_command_buffer();
            let encoder = cmd_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&llama_pipeline);
            encoder.set_buffer(0, Some(&args_buf), 0);
            encoder.set_buffer(1, Some(&weights_buf), 0);
            encoder.set_buffer(2, Some(&input_buf), 0);
            encoder.set_buffer(3, Some(&llama_output_buf), 0);
            encoder.dispatch_thread_groups(thread_groups, thread_group_size);
            encoder.end_encoding();
            cmd_buffer.commit();
            cmd_buffer.wait_until_completed();
        }
        let llama_time = start.elapsed().as_micros() as f64 / iterations as f64;

        // Benchmark self-written
        let start = Instant::now();
        for _ in 0..iterations {
            let cmd_buffer = queue.new_command_buffer();
            let encoder = cmd_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&my_pipeline);
            encoder.set_buffer(0, Some(&args_buf), 0);
            encoder.set_buffer(1, Some(&weights_buf), 0);
            encoder.set_buffer(2, Some(&input_buf), 0);
            encoder.set_buffer(3, Some(&my_output_buf), 0);
            encoder.dispatch_thread_groups(thread_groups, thread_group_size);
            encoder.end_encoding();
            cmd_buffer.commit();
            cmd_buffer.wait_until_completed();
        }
        let my_time = start.elapsed().as_micros() as f64 / iterations as f64;

        println!("llama.cpp:   {:.2} µs", llama_time);
        println!("self-written: {:.2} µs", my_time);
        println!(
            "Ratio:        {:.1}% of llama.cpp",
            llama_time / my_time * 100.0
        );
    } else {
        println!("✗ Correctness test failed");
        println!("First 10 results:");
        for i in 0..10 {
            println!(
                "  [{}] llama.cpp={:.6}, self={:.6}, err={:.6}",
                i,
                llama_output[i],
                my_output[i],
                (my_output[i] - llama_output[i]).abs()
            );
        }
    }
}
