// Compare llama.cpp shader vs self-written shader
// Run: cargo run --bin compare_q4_k_shaders --release

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

fn main() {
    let m = 4096usize;
    let k = 4096usize;
    let iterations = 10000;

    println!("=== Q4_K Shader Comparison ===");
    println!("M={}, K={}, iterations={}\n", m, k, iterations);

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Define the Q4_K kernel from llama.cpp directly
    let llama_shader = r#"
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define N_SIMDWIDTH 32
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct {
    half  d;
    half  dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

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

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12  [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2    [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3    [[function_constant(FC_MUL_MV + 4)]];

template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[16];
    float yh[16];

    float sumf[nr0]={0.f};

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01/2;
            sc += args.nb01/2;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

    let compile_options = CompileOptions::new();
    let llama_lib =
        match device.new_library_with_source(llama_shader, &compile_options) {
            Ok(lib) => {
                println!("llama.cpp shader compiled successfully");
                lib
            }
            Err(e) => {
                println!("llama.cpp shader compilation failed: {:?}", e);
                return;
            }
        };

    // Load self-written shader
    let my_source = include_str!("../../shaders/mul_mv_q4_k.metal");
    let my_lib =
        match device.new_library_with_source(my_source, &compile_options) {
            Ok(lib) => {
                println!("Self-written shader compiled successfully");
                lib
            }
            Err(e) => {
                println!("Self-written shader compilation failed: {:?}", e);
                return;
            }
        };

    // Create pipelines
    let cv = create_function_constants(2, 1, 1, 1);

    let llama_kernel = llama_lib
        .get_function("kernel_mul_mv_q4_K_f32", Some(cv.clone()))
        .expect("Failed to get llama.cpp kernel");
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_kernel)
        .expect("Failed to create llama.cpp pipeline");

    let my_kernel = my_lib
        .get_function("kernel_mul_mv_q4_K_f32", Some(cv))
        .expect("Failed to get self-written kernel");
    let my_pipeline = device
        .new_compute_pipeline_state_with_function(&my_kernel)
        .expect("Failed to create self-written pipeline");

    // Prepare data
    let weights = generate_weights(m, k);
    let input = generate_input(k);

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

    // Thread group configuration
    const NSG: u64 = 2;
    const NR0: u64 = 2;
    const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

    let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
    let thread_group_count = MTLSize {
        width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
        height: 1,
        depth: 1,
    };

    // Test both shaders alternately to eliminate thermal bias
    let output_llama = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
    let output_mine = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);

    // Extended warmup for both
    for _ in 0..100 {
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&llama_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_llama), 0);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&my_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_mine), 0);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Benchmark both alternately, 5 rounds each
    let mut llama_times = Vec::new();
    let mut mine_times = Vec::new();
    let mut llama_output = vec![0.0f32; 5];
    let mut mine_output = vec![0.0f32; 5];

    for round in 0..5 {
        // llama.cpp
        let start = Instant::now();
        for _ in 0..iterations {
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&llama_pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const c_void,
            );
            encoder.set_buffer(1, Some(&weights_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&output_llama), 0);
            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        let elapsed_us =
            start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
        llama_times.push(elapsed_us);

        // Self-written
        let start = Instant::now();
        for _ in 0..iterations {
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&my_pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const c_void,
            );
            encoder.set_buffer(1, Some(&weights_buffer), 0);
            encoder.set_buffer(2, Some(&input_buffer), 0);
            encoder.set_buffer(3, Some(&output_mine), 0);
            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        let elapsed_us =
            start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
        mine_times.push(elapsed_us);

        println!(
            "Round {}: llama.cpp {:.2} us, self-written {:.2} us",
            round + 1,
            llama_times.last().unwrap(),
            mine_times.last().unwrap()
        );
    }

    // Get outputs
    let llama_ptr = output_llama.contents() as *const f32;
    llama_output = (0..5).map(|i| unsafe { *llama_ptr.add(i) }).collect();
    let mine_ptr = output_mine.contents() as *const f32;
    mine_output = (0..5).map(|i| unsafe { *mine_ptr.add(i) }).collect();

    let llama_best = llama_times.iter().cloned().fold(f64::MAX, f64::min);
    let mine_best = mine_times.iter().cloned().fold(f64::MAX, f64::min);
    let llama_avg = llama_times.iter().sum::<f64>() / llama_times.len() as f64;
    let mine_avg = mine_times.iter().sum::<f64>() / mine_times.len() as f64;

    let llama_gflops = (2.0 * m as f64 * k as f64) / (llama_best * 1e-6) / 1e9;
    let mine_gflops = (2.0 * m as f64 * k as f64) / (mine_best * 1e-6) / 1e9;

    println!("\n=== llama.cpp shader ===");
    println!("output[0..4]: {:?}", llama_output);
    println!("Best time: {:.2} us, Avg time: {:.2} us", llama_best, llama_avg);
    println!("GFLOPS: {:.2}", llama_gflops);

    println!("\n=== Self-written shader ===");
    println!("output[0..4]: {:?}", mine_output);
    println!("Best time: {:.2} us, Avg time: {:.2} us", mine_best, mine_avg);
    println!("GFLOPS: {:.2}", mine_gflops);

    println!("\n=== Comparison ===");
    println!("Ratio: {:.2}%", mine_gflops / llama_gflops * 100.0);
}
