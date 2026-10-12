// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Verify IQ quantization MV kernels: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS

use half::f16;
use metal::*;
use std::ffi::c_void;

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockIQ1S {
    d: u16,
    qs: [u8; 32],
    qh: [u16; 8],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockIQ1M {
    qs: [u8; 32],
    qh: [u8; 16],
    scales: [u16; 4],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockIQ2XXS {
    d: u16,
    qs: [u16; 32],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockIQ2XS {
    d: u16,
    qs: [u16; 32],
    scales: [u8; 8],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockIQ2S {
    d: u16,
    qs: [u8; 32],
    qh: [u8; 8],
    signs: [u8; 32],
    scales: [u8; 8],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockIQ3XXS {
    d: u16,
    qs: [u8; 32],
    qh: [u16; 16],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockIQ3S {
    d: u16,
    qs: [u8; 32],
    qh: [u8; 8],
    signs: [u8; 16],
    scales: [u8; 4],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockIQ4NL {
    d: u16,
    qs: [u8; 16], // QK4_NL/2 = 32/2 = 16
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockIQ4XS {
    d: u16,
    qs: [u8; 32],
    scales_h: u16,
    scales_l: [u8; 8],
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

// Generate weights for each format
fn generate_iq1_s_weights(m: usize, k: usize) -> Vec<BlockIQ1S> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 32];
            let mut qh = [0u16; 8];
            for j in 0..32 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..8 {
                qh[j] = ((i * 19 + j * 7) % 65536) as u16;
            }
            BlockIQ1S {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
                qh,
            }
        })
        .collect()
}

fn generate_iq1_m_weights(m: usize, k: usize) -> Vec<BlockIQ1M> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 32];
            let mut qh = [0u8; 16];
            let mut scales = [0u16; 4];
            for j in 0..32 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..16 {
                qh[j] = ((i * 19 + j * 7) % 256) as u8;
            }
            for j in 0..4 {
                scales[j] = ((i * 23 + j * 11) % 65536) as u16;
            }
            BlockIQ1M { qs, qh, scales }
        })
        .collect()
}

fn generate_iq2_xxs_weights(m: usize, k: usize) -> Vec<BlockIQ2XXS> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u16; 32];
            for j in 0..32 {
                qs[j] = ((i * 17 + j * 13) % 65536) as u16;
            }
            BlockIQ2XXS {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_iq2_xs_weights(m: usize, k: usize) -> Vec<BlockIQ2XS> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u16; 32];
            let mut scales = [0u8; 8];
            for j in 0..32 {
                qs[j] = ((i * 17 + j * 13) % 65536) as u16;
            }
            for j in 0..8 {
                scales[j] = ((i * 19 + j * 7) % 256) as u8;
            }
            BlockIQ2XS {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
                scales,
            }
        })
        .collect()
}

fn generate_iq2_s_weights(m: usize, k: usize) -> Vec<BlockIQ2S> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 32];
            let mut qh = [0u8; 8];
            let mut signs = [0u8; 32];
            let mut scales = [0u8; 8];
            for j in 0..32 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
                signs[j] = ((i * 31 + j * 17) % 256) as u8;
            }
            for j in 0..8 {
                qh[j] = ((i * 19 + j * 7) % 256) as u8;
                scales[j] = ((i * 23 + j * 11) % 256) as u8;
            }
            BlockIQ2S {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
                qh,
                signs,
                scales,
            }
        })
        .collect()
}

fn generate_iq3_xxs_weights(m: usize, k: usize) -> Vec<BlockIQ3XXS> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 32];
            let mut qh = [0u16; 16];
            for j in 0..32 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..16 {
                qh[j] = ((i * 19 + j * 7) % 65536) as u16;
            }
            BlockIQ3XXS {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
                qh,
            }
        })
        .collect()
}

fn generate_iq3_s_weights(m: usize, k: usize) -> Vec<BlockIQ3S> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 32];
            let mut qh = [0u8; 8];
            let mut signs = [0u8; 16];
            let mut scales = [0u8; 4];
            for j in 0..32 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..8 {
                qh[j] = ((i * 19 + j * 7) % 256) as u8;
            }
            for j in 0..16 {
                signs[j] = ((i * 31 + j * 17) % 256) as u8;
            }
            for j in 0..4 {
                scales[j] = ((i * 23 + j * 11) % 256) as u8;
            }
            BlockIQ3S {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
                qh,
                signs,
                scales,
            }
        })
        .collect()
}

fn generate_iq4_nl_weights(m: usize, k: usize) -> Vec<BlockIQ4NL> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockIQ4NL {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_iq4_xs_weights(m: usize, k: usize) -> Vec<BlockIQ4XS> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 32];
            let mut scales_l = [0u8; 8];
            for j in 0..32 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..8 {
                scales_l[j] = ((i * 19 + j * 7) % 256) as u8;
            }
            BlockIQ4XS {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
                scales_h: ((i * 23) % 65536) as u16,
                scales_l,
            }
        })
        .collect()
}

// IQ4_NL kernel from llama.cpp (uses shared memory for kvalues)
const LLAMA_IQ4_NL_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define QK4_NL 32
#define N_SIMDWIDTH 32
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct { half d; uint8_t qs[16]; } block_iq4_nl;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];

constexpr constant static float kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

template<int NR0, typename args_t>
void kernel_mul_mm_iq4_nl_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    threadgroup float * shmem_f32 = (threadgroup float *) shmem;
    const int r0 = tgpig.x; const int r1 = tgpig.y; const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * NR0;
    const uint i12 = im%FC_mul_mv_ne12; const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);
    const int nb = args.ne00/QK4_NL;
    const int ns01 = args.nb01/args.nb00;
    const short ix = tiisg/2;
    const short it = tiisg%2;
    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 yl[4];
    float sumf[NR0]={0.f};
    device const float * yb = y + ix*QK4_NL + it*8;
    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;
    float4 qf1, qf2;
    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];
        for (short row = 0; row < NR0; row++) {
            device const block_iq4_nl & xb = x[row*ns01 + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);
            float4 acc1 = {0.f}, acc2 = {0.f};
            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;
            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;
            acc1 += acc2;
            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }
        yb += 16 * QK4_NL;
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) { dst_f32[first_row + row] = sum_all; }
    }
}
[[host_name("kernel_mul_mm_iq4_nl_f32")]]
kernel void kernel_mul_mm_iq4_nl_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup char shmem[64];
    kernel_mul_mm_iq4_nl_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}
"#;

fn main() {
    println!("=== IQ Format MV Kernel Verification ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let compile_options = CompileOptions::new();

    // Compile llama.cpp IQ4_NL kernel
    let llama_iq4_nl_lib = device
        .new_library_with_source(LLAMA_IQ4_NL_KERNEL, &compile_options)
        .expect("Failed to compile llama.cpp IQ4_NL");

    // Compile self-written template
    let shader_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let my_source =
        std::fs::read_to_string(shader_dir.join("gemm.metal")).unwrap();
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

    // Test IQ4_NL first (simplest)
    test_format(
        "IQ4_NL",
        &device,
        &queue,
        &llama_iq4_nl_lib,
        &my_lib,
        &cv,
        m,
        k,
        &input_buffer,
        output_size,
        tg_size,
        tg_count,
        generate_iq4_nl_weights(m, k),
        2,
        32,
    );

    // Other IQ formats require more complex kernel extraction
    println!("\n=== Other IQ formats pending ===");
    println!("IQ1_S, IQ1_M: require shared memory kernels");
    println!("IQ2_XXS, IQ2_XS, IQ2_S: require shared memory kernels");
    println!("IQ3_XXS, IQ3_S: require shared memory kernels");
    println!("IQ4_XS: requires shared memory kernel");
}

fn test_format<T: Copy>(
    name: &str,
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
    weights: Vec<T>,
    nr0: i32,
    qk: usize,
) {
    println!("=== {} ===", name);

    let block_size = std::mem::size_of::<T>();
    let nb = k / qk;
    let nb01 = (block_size * nb) as u64;
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * block_size) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let args = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: block_size as u64,
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
        nr0,
        r2: 1,
        r3: 1,
    };

    let kernel_name = format!("kernel_mul_mm_{}_f32", name.to_lowercase());
    let llama_kernel =
        llama_lib.get_function(&kernel_name, Some(cv.clone())).unwrap();
    let my_kernel =
        my_lib.get_function(&kernel_name, Some(cv.clone())).unwrap();

    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_kernel).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_kernel).unwrap();

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

    println!("Correctness: {}/{} ({}%)", correct, m, correct * 100 / m);

    let llama_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *llama_ptr.add(i) }).collect();
    let my_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *my_ptr.add(i) }).collect();
    println!("llama.cpp: {:?}", llama_sample);
    println!("Self-written: {:?}", my_sample);

    if correct == m {
        println!("✅ {}: 100% correct", name);
    } else {
        println!("⚠️ {}: {}% correct", name, correct * 100 / m);
    }
}
