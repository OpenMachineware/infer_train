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
    println!("=== Q3_K MV: llama.cpp vs self-written ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

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
void kernel_gemm_q3_K_f32_impl(
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

[[host_name("kernel_gemm_q3_K_f32")]]
kernel void kernel_gemm_q3_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_gemm_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
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
    let my_source = include_str!("../src/ops/metal/gemm.metal");
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
        .get_function("kernel_gemm_q3_K_f32", Some(cv.clone()))
        .unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_func).unwrap();

    let my_func =
        my_lib.get_function("kernel_gemm_q3_K_f32", Some(cv.clone())).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_func).unwrap();

    println!("Pipelines created.\n");

    // Test dimensions
    let m = 1024;
    let k = 4096;
    let n = 1;
    let nb = k / 256;

    // Generate Q3_K weights (same pattern for both)
    let weights: Vec<BlockQ3K> = (0..m * nb)
        .map(|i| {
            let mut hmask = [0u8; 32];
            let mut qs = [0u8; 64];
            let mut scales = [0u8; 12];
            for j in 0..32 {
                hmask[j] = ((i * 17 + j * 23) % 256) as u8;
            }
            for j in 0..64 {
                qs[j] = ((i * 13 + j * 7) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 31 + j * 11) % 128) as u8;
            }
            BlockQ3K {
                hmask,
                qs,
                scales,
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            }
        })
        .collect();

    let input: Vec<f32> = (0..k).map(|i| ((i % 100) as f32) * 0.01).collect();

    // Create buffers
    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const _,
        (m * nb * std::mem::size_of::<BlockQ3K>()) as u64,
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
        nb01: (nb * std::mem::size_of::<BlockQ3K>()) as u64,
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
        nr0: 2,
        r2: 1,
        r3: 1,
    };

    let args_buf = device.new_buffer_with_data(
        &args as *const _ as *const _,
        std::mem::size_of::<MulMVArgs>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Run llama.cpp kernel
    let cmd_buffer = queue.new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&llama_pipeline);
    encoder.set_buffer(0, Some(&args_buf), 0);
    encoder.set_buffer(1, Some(&weights_buf), 0);
    encoder.set_buffer(2, Some(&input_buf), 0);
    encoder.set_buffer(3, Some(&llama_output_buf), 0);

    let thread_groups = MTLSize::new(1, n as u64, 1);
    let thread_group_size = MTLSize::new(32, 1, 1);
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
