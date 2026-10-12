// Compare Q3_K MV kernel vs llama.cpp
use half::f16;
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

#[repr(C, packed)]
#[derive(Copy, Clone)]
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

fn generate_weights(m: usize, k: usize) -> Vec<BlockQ3K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut hmask = [0u8; 32];
            let mut qs = [0u8; 64];
            let mut scales = [0u8; 12];
            for j in 0..32 {
                hmask[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            for j in 0..64 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 23 + j * 7) % 256) as u8;
            }
            BlockQ3K {
                hmask,
                qs,
                scales,
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            }
        })
        .collect()
}

fn generate_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

const LLAMA_Q3K_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; half d; } block_q3_K;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];

template<int nr0, typename args_t>
void kernel_gemm_q3_K_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, threadgroup char *, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short ix = tiisg/4; const short ir = tiisg%4;
    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x; const int r1 = tgpig.y; const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%FC_mul_mv_ne12; const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;
    device const block_q3_K * x = (device const block_q3_K *)(src0 + offset0);
    device const float * y = (device const float *)(src1 + offset1);
    float yl[32]; float sumf[nr0]={0.f};
    device const float * y4 = y + 128*ix + 8*ir;
    for (int ib = ix; ib < nb; ib += 8) {
        device const float4 * y_vec = (device const float4 *)y4;
        FOR_UNROLL (short i = 0; i < 8; ++i) yl[i] = y_vec[i].x, yl[i+8] = y_vec[i].y, yl[i+16] = y_vec[i].z, yl[i+24] = y_vec[i].w;
        y4 += 8*QK_K;
        device const block_q3_K * xb = x + ib;
        FOR_UNROLL (short row = 0; row < nr0; row++) {
            const float d_all = (float)xb[row].d;
            device const uint16_t * a = (device const uint16_t *)xb[row].scales;
            device const uint16_t * q = (device const uint16_t *)xb[row].qs + 16*ir;
            device const uint16_t * h = (device const uint16_t *)xb[row].hmask;
            float acc = 0.f;
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                const uint16_t qi = q[i];
                const uint16_t hi = h[i];
                uint16_t m0 = 1, m1 = 2, m2 = 4, m3 = 8;
                FOR_UNROLL (short j = 0; j < 8; ++j) {
                    uint16_t sc = a[j/2] >> (8*(j%2));
                    float d = d_all * (sc & 0xF);
                    float m = d_all * (sc >> 4);
                    uint16_t q0 = (qi >> (2*j)) & 0x3;
                    uint16_t q1 = (qi >> (2*j+16)) & 0x3;
                    uint16_t q2 = (qi >> (2*j+32)) & 0x3;
                    uint16_t q3 = (qi >> (2*j+48)) & 0x3;
                    int8_t v0 = ((hi & m0) ? 4 : 0) + q0 - ((hi & m0) ? 0 : 4);
                    int8_t v1 = ((hi & m1) ? 4 : 0) + q1 - ((hi & m1) ? 0 : 4);
                    int8_t v2 = ((hi & m2) ? 4 : 0) + q2 - ((hi & m2) ? 0 : 4);
                    int8_t v3 = ((hi & m3) ? 4 : 0) + q3 - ((hi & m3) ? 0 : 4);
                    acc += d * (v0*yl[8*j+0] + v1*yl[8*j+1] + v2*yl[8*j+2] + v3*yl[8*j+3]);
                    m0 <<= 1; m1 <<= 1; m2 <<= 1; m3 <<= 1;
                }
            }
            sumf[row] += acc;
        }
    }
    device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row + row] = sum_all;
    }
}
[[host_name("kernel_gemm_q3_K_f32")]]
kernel void kernel_gemm_q3_K_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_gemm_q3_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

fn main() {
    println!("=== Q3_K Correctness & Performance ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    let llama_lib = device
        .new_library_with_source(LLAMA_Q3K_KERNEL, &compile_options)
        .expect("Failed to compile llama.cpp kernel");
    let shader_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let my_source = std::fs::read_to_string(shader_dir.join("gemm.metal"))
        .expect("Failed to read template");
    let my_lib = device
        .new_library_with_source(&my_source, &compile_options)
        .expect("Failed to compile template");

    let m = 4096usize;
    let k = 4096usize;
    let iterations = 5000;
    let warmup = 100;

    let weights = generate_weights(m, k);
    let input = generate_input(k);

    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * 110) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cv = create_cv();
    let llama_kernel = llama_lib
        .get_function("kernel_gemm_q3_K_f32", Some(cv.clone()))
        .unwrap();
    let my_kernel =
        my_lib.get_function("kernel_gemm_q3_K_f32", Some(cv.clone())).unwrap();

    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_kernel).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_kernel).unwrap();

    let nb = k / 256;
    let nb01 = (110 * nb) as u64;
    let args = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 110,
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

    let output_size = m * 4;
    let tg_size = MTLSize { width: 32, height: 2, depth: 1 };
    let tg_count = MTLSize { width: (m as u64 + 3) / 4, height: 1, depth: 1 };

    // Warmup
    for _ in 0..warmup {
        for pipeline in [&llama_pipeline, &my_pipeline] {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            let out = device.new_buffer(
                output_size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            enc.set_compute_pipeline_state(pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buffer), 0);
            enc.set_buffer(2, Some(&input_buffer), 0);
            enc.set_buffer(3, Some(&out), 0);
            enc.dispatch_thread_groups(tg_count, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    // Benchmark
    let mut llama_time = 0u128;
    let llama_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
    for _ in 0..iterations {
        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&llama_output), 0);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        llama_time += start.elapsed().as_micros();
    }

    let mut my_time = 0u128;
    let my_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
    for _ in 0..iterations {
        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&my_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buffer), 0);
        enc.set_buffer(2, Some(&input_buffer), 0);
        enc.set_buffer(3, Some(&my_output), 0);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        my_time += start.elapsed().as_micros();
    }

    // Correctness
    let llama_ptr = llama_output.contents() as *const f32;
    let my_ptr = my_output.contents() as *const f32;
    let correct: usize = (0..m)
        .filter(|&i| {
            let lv = unsafe { *llama_ptr.add(i) };
            let mv = unsafe { *my_ptr.add(i) };
            if lv.abs() > 1e-6 {
                (lv - mv).abs() / lv.abs() < 0.01
            } else {
                (lv - mv).abs() < 1e-6
            }
        })
        .count();

    println!("Correctness: {}/{} ({}%)", correct, m, correct * 100 / m);
    println!(
        "\nllama.cpp: {:.2} us ({:.1} GFLOPS)",
        llama_time as f64 / iterations as f64,
        (2.0 * m as f64 * k as f64)
            / (llama_time as f64 / iterations as f64 * 1e-6)
            / 1e9
    );
    println!(
        "Self-written: {:.2} us ({:.1} GFLOPS)",
        my_time as f64 / iterations as f64,
        (2.0 * m as f64 * k as f64)
            / (my_time as f64 / iterations as f64 * 1e-6)
            / 1e9
    );
    println!(
        "Performance: {:.1}% of llama.cpp",
        llama_time as f64 / my_time as f64 * 100.0
    );
}
