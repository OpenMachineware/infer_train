// Direct comparison: Q2_K self-written vs llama.cpp
// Test correctness and performance

use half::f16;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ2K {
    scales: [u8; 16],
    qs: [u8; 64],
    d: u16,
    dmin: u16,
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

fn expand_includes(
    source: &str,
    shader_dir: &std::path::PathBuf,
    visited: &mut Vec<String>,
) -> String {
    let mut result = String::new();
    for line in source.lines() {
        if line.starts_with("#include \"") {
            let start = line.find('"').unwrap() + 1;
            let end = line.rfind('"').unwrap();
            let filename = &line[start..end];
            if visited.contains(&filename.to_string()) {
                continue;
            }
            visited.push(filename.to_string());
            let include_path = shader_dir.join(filename);
            if let Ok(include_content) = std::fs::read_to_string(&include_path)
            {
                let include_content =
                    include_content.replace("#pragma once", "// #pragma once");
                result.push_str(&expand_includes(
                    &include_content,
                    shader_dir,
                    visited,
                ));
            }
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

fn generate_q2k_weights(m: usize, k: usize) -> Vec<BlockQ2K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut scales = [0u8; 16];
            let mut qs = [0u8; 64];
            for j in 0..16 {
                scales[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            for j in 0..64 {
                qs[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            BlockQ2K {
                scales,
                qs,
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
            }
        })
        .collect()
}

fn generate_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

fn main() {
    println!("=== Q2_K Correctness & Performance ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // llama.cpp Q2_K kernel
    let llama_shader = r#"
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct {
    uint8_t scales[16];
    uint8_t qs[64];
    half  d;
    half  dmin;
} block_q2_K;

typedef struct {
    int32_t ne00; int32_t ne01; int32_t ne02;
    uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
    int32_t ne10; int32_t ne11; int32_t ne12;
    uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
    int32_t ne0; int32_t ne1;
    int32_t nr0;
    int16_t r2;
    int16_t r3;
} ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12  [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2    [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3    [[function_constant(FC_MUL_MV + 4)]];

template<int nr0, typename args_t>
void kernel_mul_mm_q2_K_f32_impl(
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

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mm_q2_K_f32")]]
kernel void kernel_mul_mm_q2_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mm_q2_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

    // Compile llama.cpp
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(true);
    let llama_lib = device
        .new_library_with_source(llama_shader, &compile_options)
        .expect("Failed to compile llama.cpp");

    // Compile self-written
    let shader_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let my_source = std::fs::read_to_string(shader_dir.join("gemm.metal"))
        .expect("Failed to read");
    let mut visited = Vec::new();
    let my_shader = expand_includes(&my_source, &shader_dir, &mut visited);
    let my_lib = device
        .new_library_with_source(&my_shader, &compile_options)
        .expect("Failed to compile self-written");

    // Test parameters
    let m = 4096usize;
    let k = 4096usize;
    let iterations = 10000;

    let weights = generate_q2k_weights(m, k);
    let input = generate_input(k);

    let nb = k / 256;
    let nb01 = (nb * std::mem::size_of::<BlockQ2K>()) as u64; // 84 bytes per block
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
        nr0: 4, // Q2_K uses nr0=4
        r2: 1,
        r3: 1,
    };

    let weights_size = weights.len() * std::mem::size_of::<BlockQ2K>();
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

    let cv = create_function_constants(2, 1, 1, 1);

    // Get kernels
    let llama_kernel = llama_lib
        .get_function("kernel_mul_mm_q2_K_f32", Some(cv.clone()))
        .expect("Failed to get llama.cpp kernel");
    let my_kernel = my_lib
        .get_function("kernel_mul_mm_q2_K_f32", Some(cv))
        .expect("Failed to get self-written kernel");

    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_kernel)
        .expect("Failed to create llama.cpp pipeline");
    let my_pipeline = device
        .new_compute_pipeline_state_with_function(&my_kernel)
        .expect("Failed to create self-written pipeline");

    const NSG: u64 = 2;
    const ROWS_PER_THREADGROUP: u64 = NSG * 4;
    let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
    let thread_group_count = MTLSize {
        width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
        height: 1,
        depth: 1,
    };

    // Warmup and benchmark llama.cpp
    let llama_output_buffer = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
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
        encoder.set_buffer(3, Some(&llama_output_buffer), 0);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    let mut llama_best_us = f64::MAX;
    for _ in 0..3 {
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
            encoder.set_buffer(3, Some(&llama_output_buffer), 0);
            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        let elapsed_us =
            start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
        if elapsed_us < llama_best_us {
            llama_best_us = elapsed_us;
        }
    }

    // Warmup and benchmark self-written
    let my_output_buffer = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
    for _ in 0..100 {
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
        encoder.set_buffer(3, Some(&my_output_buffer), 0);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    let mut my_best_us = f64::MAX;
    for _ in 0..3 {
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
            encoder.set_buffer(3, Some(&my_output_buffer), 0);
            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        let elapsed_us =
            start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
        if elapsed_us < my_best_us {
            my_best_us = elapsed_us;
        }
    }

    // Compare outputs
    let llama_ptr = llama_output_buffer.contents() as *const f32;
    let my_ptr = my_output_buffer.contents() as *const f32;

    let mut max_diff = 0.0f32;
    let mut rel_diff_sum = 0.0f32;
    let correct_count: usize = (0..m)
        .filter(|&i| {
            let llama_val = unsafe { *llama_ptr.add(i) };
            let my_val = unsafe { *my_ptr.add(i) };
            let diff = (llama_val - my_val).abs();
            let rel_diff = if llama_val.abs() > 1e-6 {
                diff / llama_val.abs()
            } else {
                diff
            };
            max_diff = max_diff.max(diff);
            rel_diff_sum += rel_diff;
            rel_diff < 0.01
        })
        .count();

    let llama_gflops =
        (2.0 * m as f64 * k as f64) / (llama_best_us * 1e-6) / 1e9;
    let my_gflops = (2.0 * m as f64 * k as f64) / (my_best_us * 1e-6) / 1e9;
    let ratio = (my_gflops / llama_gflops) * 100.0;

    println!("Block size: {} bytes", std::mem::size_of::<BlockQ2K>());
    println!();
    println!(
        "Correctness: {}/{} ({}%)",
        correct_count,
        m,
        (correct_count * 100 / m)
    );
    println!("Max diff: {}", max_diff);
    println!("Avg rel diff: {}", rel_diff_sum / m as f32);
    println!();
    println!("llama.cpp: {:.2} us ({:.1} GFLOPS)", llama_best_us, llama_gflops);
    println!("Self-written: {:.2} us ({:.1} GFLOPS)", my_best_us, my_gflops);
    println!("Performance: {:.1}% of llama.cpp", ratio);
}
