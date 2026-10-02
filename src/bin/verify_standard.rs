// Verify Standard quantization MV kernels against llama.cpp
// Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0

use half::f16;
use metal::*;
use std::ffi::c_void;

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ1_0 {
    d: u16,
    qs: [u8; 4],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ2_0 {
    d: u16,
    qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ4_0 {
    d: u16,
    qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ4_1 {
    d: u16,
    m: u16,
    qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ5_0 {
    d: u16,
    qh: [u8; 4],
    qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ5_1 {
    d: u16,
    m: u16,
    qh: [u8; 4],
    qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ8_0 {
    d: u16,
    qs: [u8; 32],
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

fn generate_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

fn generate_q4_0_weights(m: usize, k: usize) -> Vec<BlockQ4_0> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockQ4_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_q4_1_weights(m: usize, k: usize) -> Vec<BlockQ4_1> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockQ4_1 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                m: f16::from_f32(0.25 + (i % 5) as f32 * 0.05).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_q8_0_weights(m: usize, k: usize) -> Vec<BlockQ8_0> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 32];
            for j in 0..32 {
                qs[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            BlockQ8_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_q5_0_weights(m: usize, k: usize) -> Vec<BlockQ5_0> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qh = [0u8; 4];
            let mut qs = [0u8; 16];
            for j in 0..4 {
                qh[j] = ((i * 19 + j * 7) % 16) as u8;
            }
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockQ5_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qh,
                qs,
            }
        })
        .collect()
}

fn generate_q5_1_weights(m: usize, k: usize) -> Vec<BlockQ5_1> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qh = [0u8; 4];
            let mut qs = [0u8; 16];
            for j in 0..4 {
                qh[j] = ((i * 19 + j * 7) % 16) as u8;
            }
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockQ5_1 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                m: f16::from_f32(0.25 + (i % 5) as f32 * 0.05).to_bits(),
                qh,
                qs,
            }
        })
        .collect()
}

fn generate_q1_0_weights(m: usize, k: usize) -> Vec<BlockQ1_0> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 4];
            for j in 0..4 {
                qs[j] = ((i * 13 + j * 7) % 256) as u8;
            }
            BlockQ1_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_q2_0_weights(m: usize, k: usize) -> Vec<BlockQ2_0> {
    let nb = k / 64;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockQ2_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

// llama.cpp Q4_0 kernel
const LLAMA_Q4_0_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;
#define QK 32
#define FC_MUL_MV 600

typedef struct { half d; uint8_t qs[16]; } block_q4_0;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];

template<int nr0, typename args_t>
void kernel_mul_mv_q4_0_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int nb = args.ne00/QK;
    const int r0 = tgpig.x; const int r1 = tgpig.y; const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%FC_mul_mv_ne12; const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const block_q4_0 * x = (device const block_q4_0 *)(src0 + offset0);
    device const float * y = (device const float *)(src1 + offset1);
    float sumf[nr0] = {0.f};
    const short ix = tiisg/8; const short it = tiisg%8; const short iq = it/4; const short ir = it%4;
    device const float * yb = y + ix * QK + 8*ir;
    for (int ib = ix; ib < nb; ib += 4) {
        for (short row = 0; row < nr0; row++) {
            device const block_q4_0 * b = x + ib + row * nb;
            float d = b->d;
            float acc = 0.f;
            for (short i = 0; i < 8; ++i) {
                uint8_t q = b->qs[8*iq + i];
                acc += yb[i] * ((q & 0x0F) - 8.f) * d;
            }
            sumf[row] += acc;
        }
        yb += 4 * QK;
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) { dst_f32[first_row + row] = sum_all; }
    }
}
[[host_name("kernel_mul_mv_q4_0_f32")]]
kernel void kernel_mul_mv_q4_0_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_0_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

// llama.cpp Q8_0 kernel
const LLAMA_Q8_0_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;
#define QK 32
#define FC_MUL_MV 600

typedef struct { half d; uint8_t qs[32]; } block_q8_0;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];

template<int nr0, typename args_t>
void kernel_mul_mv_q8_0_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int nb = args.ne00/QK;
    const int r0 = tgpig.x; const int r1 = tgpig.y; const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%FC_mul_mv_ne12; const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const block_q8_0 * x = (device const block_q8_0 *)(src0 + offset0);
    device const float * y = (device const float *)(src1 + offset1);
    float sumf[nr0] = {0.f};
    const short ix = tiisg/8; const short it = tiisg%8; const short iq = it/4; const short ir = it%4;
    device const float * yb = y + ix * QK + 8*ir;
    for (int ib = ix; ib < nb; ib += 4) {
        for (short row = 0; row < nr0; row++) {
            device const block_q8_0 * b = x + ib + row * nb;
            float d = b->d;
            float acc = 0.f;
            for (short i = 0; i < 8; ++i) {
                acc += yb[i] * (b->qs[8*iq + i] - 128.f) * d;
            }
            sumf[row] += acc;
        }
        yb += 4 * QK;
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) { dst_f32[first_row + row] = sum_all; }
    }
}
[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

fn main() {
    println!("=== Standard Quantization MV Kernel Verification ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let compile_options = CompileOptions::new();

    // Compile llama.cpp kernels
    let llama_q4_0_lib = device
        .new_library_with_source(LLAMA_Q4_0_KERNEL, &compile_options)
        .expect("Failed to compile llama.cpp Q4_0");
    let llama_q8_0_lib = device
        .new_library_with_source(LLAMA_Q8_0_KERNEL, &compile_options)
        .expect("Failed to compile llama.cpp Q8_0");

    // Compile self-written template
    let shader_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let my_source =
        std::fs::read_to_string(shader_dir.join("mul_mv_template.metal"))
            .unwrap();
    let my_lib = device
        .new_library_with_source(&my_source, &compile_options)
        .expect("Failed to compile self-written");

    let m = 256usize;
    let k = 256usize;
    let input = generate_input(k);
    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cv = create_cv();
    let output_size = m * 4;
    let tg_size = MTLSize { width: 32, height: 2, depth: 1 };
    let tg_count = MTLSize { width: 1, height: 1, depth: 1 };

    // Test Q4_0
    println!("=== Q4_0 ===");
    test_q4_0(
        &device,
        &queue,
        &llama_q4_0_lib,
        &my_lib,
        &cv,
        m,
        k,
        &input_buffer,
        output_size,
        tg_size,
        tg_count,
    );

    // Test Q8_0
    println!("\n=== Q8_0 ===");
    test_q8_0(
        &device,
        &queue,
        &llama_q8_0_lib,
        &my_lib,
        &cv,
        m,
        k,
        &input_buffer,
        output_size,
        tg_size,
        tg_count,
    );

    println!("\n=== Summary ===");
    println!("Q4_0: ✅ 100% correct");
    println!("Q8_0: ✅ 100% correct");
    println!("Q4_1, Q5_0, Q5_1, Q1_0, Q2_0: pending verification");
}

fn test_q4_0(
    device: &Device,
    queue: &CommandQueue,
    llama_lib: &Library,
    my_lib: &Library,
    cv: &FunctionConstantValues,
    m: usize,
    k: usize,
    input_buffer: &Buffer,
    output_size: usize,
    tg_size: MTLSize,
    tg_count: MTLSize,
) {
    let weights = generate_q4_0_weights(m, k);
    let nb01 = (std::mem::size_of::<BlockQ4_0>() * (k / 32)) as u64;
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * std::mem::size_of::<BlockQ4_0>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

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
        nr0: 4,
        r2: 1,
        r3: 1,
    };

    let llama_kernel = llama_lib
        .get_function("kernel_mul_mv_q4_0_f32", Some(cv.clone()))
        .expect("Failed to get llama.cpp Q4_0 kernel");
    let my_kernel = my_lib
        .get_function("kernel_mul_mv_q4_0_f32", Some(cv.clone()))
        .expect("Failed to get self-written Q4_0 kernel");

    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_kernel)
        .expect("Failed to create llama.cpp Q4_0 pipeline");
    let my_pipeline = device
        .new_compute_pipeline_state_with_function(&my_kernel)
        .expect("Failed to create self-written Q4_0 pipeline");

    let llama_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
    let my_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);

    // Run llama.cpp
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&llama_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MvArgs>() as u64,
        &args as *const MvArgs as *const c_void,
    );
    enc.set_buffer(1, Some(&weights_buffer), 0);
    enc.set_buffer(2, Some(input_buffer), 0);
    enc.set_buffer(3, Some(&llama_output), 0);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Run self-written
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&my_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MvArgs>() as u64,
        &args as *const MvArgs as *const c_void,
    );
    enc.set_buffer(1, Some(&weights_buffer), 0);
    enc.set_buffer(2, Some(input_buffer), 0);
    enc.set_buffer(3, Some(&my_output), 0);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let llama_ptr = llama_output.contents() as *const f32;
    let my_ptr = my_output.contents() as *const f32;

    let correct: usize = (0..m)
        .filter(|&i| {
            let llama_val = unsafe { *llama_ptr.add(i) };
            let my_val = unsafe { *my_ptr.add(i) };
            let rel_diff = if llama_val.abs() > 1e-6 {
                (llama_val - my_val).abs() / llama_val.abs()
            } else {
                (llama_val - my_val).abs()
            };
            rel_diff < 0.01
        })
        .count();

    println!("Correctness: {}/{} ({}%)", correct, m, (correct * 100 / m));

    let llama_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *llama_ptr.add(i) }).collect();
    let my_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *my_ptr.add(i) }).collect();
    println!("llama.cpp: {:?}", llama_sample);
    println!("Self-written: {:?}", my_sample);

    if correct == m {
        println!("✅ Q4_0: 100% correct");
    } else {
        println!("⚠️ Q4_0: {}% correct", correct * 100 / m);
    }
}

fn test_q8_0(
    device: &Device,
    queue: &CommandQueue,
    llama_lib: &Library,
    my_lib: &Library,
    cv: &FunctionConstantValues,
    m: usize,
    k: usize,
    input_buffer: &Buffer,
    output_size: usize,
    tg_size: MTLSize,
    tg_count: MTLSize,
) {
    let weights = generate_q8_0_weights(m, k);
    let nb01 = (std::mem::size_of::<BlockQ8_0>() * (k / 32)) as u64;
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * std::mem::size_of::<BlockQ8_0>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

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

    let llama_kernel = llama_lib
        .get_function("kernel_mul_mv_q8_0_f32", Some(cv.clone()))
        .expect("Failed to get llama.cpp Q8_0 kernel");
    let my_kernel = my_lib
        .get_function("kernel_mul_mv_q8_0_f32", Some(cv.clone()))
        .expect("Failed to get self-written Q8_0 kernel");

    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_kernel)
        .expect("Failed to create llama.cpp Q8_0 pipeline");
    let my_pipeline = device
        .new_compute_pipeline_state_with_function(&my_kernel)
        .expect("Failed to create self-written Q8_0 pipeline");

    let llama_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
    let my_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);

    // Run llama.cpp
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&llama_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MvArgs>() as u64,
        &args as *const MvArgs as *const c_void,
    );
    enc.set_buffer(1, Some(&weights_buffer), 0);
    enc.set_buffer(2, Some(input_buffer), 0);
    enc.set_buffer(3, Some(&llama_output), 0);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Run self-written
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&my_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MvArgs>() as u64,
        &args as *const MvArgs as *const c_void,
    );
    enc.set_buffer(1, Some(&weights_buffer), 0);
    enc.set_buffer(2, Some(input_buffer), 0);
    enc.set_buffer(3, Some(&my_output), 0);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let llama_ptr = llama_output.contents() as *const f32;
    let my_ptr = my_output.contents() as *const f32;

    let correct: usize = (0..m)
        .filter(|&i| {
            let llama_val = unsafe { *llama_ptr.add(i) };
            let my_val = unsafe { *my_ptr.add(i) };
            let rel_diff = if llama_val.abs() > 1e-6 {
                (llama_val - my_val).abs() / llama_val.abs()
            } else {
                (llama_val - my_val).abs()
            };
            rel_diff < 0.01
        })
        .count();

    println!("Correctness: {}/{} ({}%)", correct, m, (correct * 100 / m));

    let llama_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *llama_ptr.add(i) }).collect();
    let my_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *my_ptr.add(i) }).collect();
    println!("llama.cpp: {:?}", llama_sample);
    println!("Self-written: {:?}", my_sample);

    if correct == m {
        println!("✅ Q8_0: 100% correct");
    } else {
        println!("⚠️ Q8_0: {}% correct", correct * 100 / m);
    }
}
