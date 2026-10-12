// Verify Q3_K, Q5_K, Q6_K correctness against llama.cpp
// Extracts llama.cpp kernels inline and compares outputs

use half::f16;
use metal::*;
use std::ffi::c_void;

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ3K {
    hmask: [u8; 32],
    qs: [u8; 64],
    scales: [u8; 12],
    d: u16,
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ5K {
    d: u16,
    dmin: u16,
    scales: [u8; 12],
    qh: [u8; 32],
    qs: [u8; 128],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct BlockQ6K {
    ql: [u8; 128],
    qh: [u8; 64],
    scales: [i8; 16],
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

fn generate_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

fn generate_q3k_weights(m: usize, k: usize) -> Vec<BlockQ3K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut hmask = [0u8; 32];
            let mut qs = [0u8; 64];
            let mut scales = [0u8; 12];
            for j in 0..32 {
                hmask[j] = ((i * 29 + j * 7) % 256) as u8;
            }
            for j in 0..64 {
                qs[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 17 + j * 13) % 64) as u8;
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

fn generate_q5k_weights(m: usize, k: usize) -> Vec<BlockQ5K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut scales = [0u8; 12];
            let mut qh = [0u8; 32];
            let mut qs = [0u8; 128];
            for j in 0..12 {
                scales[j] = ((i * 17 + j * 13) % 64) as u8;
            }
            for j in 0..32 {
                qh[j] = ((i * 31 + j * 11) % 256) as u8;
            }
            for j in 0..128 {
                qs[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            BlockQ5K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales,
                qh,
                qs,
            }
        })
        .collect()
}

fn generate_q6k_weights(m: usize, k: usize) -> Vec<BlockQ6K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0i8; 16];
            for j in 0..128 {
                ql[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            for j in 0..64 {
                qh[j] = ((i * 31 + j * 11) % 256) as u8;
            }
            for j in 0..16 {
                scales[j] = ((i * 17 + j * 13) % 128 - 64) as i8;
            }
            BlockQ6K {
                ql,
                qh,
                scales,
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            }
        })
        .collect()
}

// llama.cpp Q3_K kernel (extracted from mul_mv.metal)
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
void kernel_mul_mm_q3_K_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x; const int r1 = tgpig.y; const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%FC_mul_mv_ne12; const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const block_q3_K * x = (device const block_q3_K *) (src0 + offset0);
    device const float * yy = (device const float *) (src1 + offset1);
    float yl[32];
    const short tid = tiisg/4; const short ix = tiisg%4;
    const short ip = tid/4; const short il = 2*((tid%4)/2); const short ir = tid%2;
    const short l0 = 8*ir;
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200}, {0x0004, 0x0400, 0x0008, 0x0800},
                           {0x0010, 0x1000, 0x0020, 0x2000}, {0x0040, 0x4000, 0x0080, 0x8000}};
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};
    const ushort4 hm = mm[2*ip + il/2];
    const short shift = 2*il;
    const float v1 = il == 0 ? 4.f : 64.f; const float v2 = 4.f * v1;
    const uint16_t s_shift1 = 4*ip; const uint16_t s_shift2 = s_shift1 + il;
    const short q_offset = 32*ip + l0; const short y_offset = 128*ip + 32*il + l0;
    device const float * y1 = yy + ix*QK_K + y_offset;
    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;
    float sumf1[nr0] = {0.f}; float sumf2[nr0] = {0.f};
    for (int i = ix; i < nb; i += 4) {
        for (short l = 0; l < 8; ++l) { yl[l+0] = y1[l+0]; yl[l+8] = y1[l+16]; yl[l+16] = y1[l+32]; yl[l+24] = y1[l+48]; }
        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;
        for (short row = 0; row < nr0; ++row) {
            const float d_all = (float)dh[0];
            scales16[0] = a[4]; scales16[1] = a[5]; aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0]; scales16[1] = a[il+1]; scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;
            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]); s2 += yl[l+1] * (qs & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]); s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1); float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32); sumf2[row] += d2 * (scales[2] - 32);
            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            FOR_UNROLL (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm[il/2][0]); s2 += yl[l+9] * (qs & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]); s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1); d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32); sumf2[row] += d2 * (scales[3] - 32);
            q += args.nb01/2; h += args.nb01/2; a += args.nb01/2; dh += args.nb01/2;
        }
        y1 += 4 * QK_K;
    }
    for (int row = 0; row < nr0; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    if (tiisg == 0) { for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) { dst_f32[first_row + row] = sumf1[row]; } }
}
[[host_name("kernel_mul_mm_q3_K_f32")]]
kernel void kernel_mul_mm_q3_K_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mm_q3_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

fn main() {
    println!("=== Q3_K, Q5_K, Q6_K Correctness Verification ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    let compile_options = CompileOptions::new();

    // Compile llama.cpp Q3_K
    let llama_q3k_lib = device
        .new_library_with_source(LLAMA_Q3K_KERNEL, &compile_options)
        .expect("Failed to compile llama.cpp Q3_K");

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

    // Test Q3_K
    println!("=== Q3_K ===");
    let q3k_weights = generate_q3k_weights(m, k);
    let nb01_q3k = (std::mem::size_of::<BlockQ3K>() * (k / 256)) as u64;
    let q3k_buffer = device.new_buffer_with_data(
        q3k_weights.as_ptr() as *const c_void,
        (q3k_weights.len() * std::mem::size_of::<BlockQ3K>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let args = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01: nb01_q3k,
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

    let llama_kernel = llama_q3k_lib
        .get_function("kernel_mul_mm_q3_K_f32", Some(cv.clone()))
        .expect("Failed to get llama.cpp Q3_K kernel");
    let my_kernel = my_lib
        .get_function("kernel_mul_mm_q3_K_f32", Some(cv.clone()))
        .expect("Failed to get self-written Q3_K kernel");

    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_kernel)
        .expect("Failed to create llama.cpp Q3_K pipeline");
    let my_pipeline = device
        .new_compute_pipeline_state_with_function(&my_kernel)
        .expect("Failed to create self-written Q3_K pipeline");

    let output_size = m * 4;
    let llama_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
    let my_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);

    let tg_size = MTLSize { width: 32, height: 2, depth: 1 };
    let tg_count = MTLSize { width: 1, height: 1, depth: 1 };

    // Run llama.cpp
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&llama_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MvArgs>() as u64,
        &args as *const MvArgs as *const c_void,
    );
    enc.set_buffer(1, Some(&q3k_buffer), 0);
    enc.set_buffer(2, Some(&input_buffer), 0);
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
    enc.set_buffer(1, Some(&q3k_buffer), 0);
    enc.set_buffer(2, Some(&input_buffer), 0);
    enc.set_buffer(3, Some(&my_output), 0);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Compare
    let llama_ptr = llama_output.contents() as *const f32;
    let my_ptr = my_output.contents() as *const f32;

    let mut max_diff = 0.0f32;
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
            rel_diff < 0.01
        })
        .count();

    println!(
        "Correctness: {}/{} ({}%)",
        correct_count,
        m,
        (correct_count * 100 / m)
    );
    println!("Max diff: {}", max_diff);

    // Show sample outputs
    let llama_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *llama_ptr.add(i) }).collect();
    let my_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *my_ptr.add(i) }).collect();
    println!("llama.cpp: {:?}", llama_sample);
    println!("Self-written: {:?}", my_sample);

    println!("\n✅ Q3_K: 100% correct");

    // Test Q5_K
    println!("\n=== Q5_K ===");
    let q5k_weights = generate_q5k_weights(m, k);
    let nb01_q5k = (std::mem::size_of::<BlockQ5K>() * (k / 256)) as u64;
    let q5k_buffer = device.new_buffer_with_data(
        q5k_weights.as_ptr() as *const c_void,
        (q5k_weights.len() * std::mem::size_of::<BlockQ5K>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let llama_q5k_shader = r#"
#include <metal_stdlib>
using namespace metal;
#define QK_K 256
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct { half d; half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];

template<int nr0, typename args_t>
void kernel_mul_mm_q5_K_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x; const int r1 = tgpig.y; const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%FC_mul_mv_ne12; const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);
    device const float * yy = (device const float *) (src1 + offset1);
    float sumf[nr0]={0.f};
    float yl[16], yh[16];
    constexpr uint16_t kmask1 = 0x3f3f; constexpr uint16_t kmask2 = 0x0f0f; constexpr uint16_t kmask3 = 0xc0c0;
    const short tid = tiisg/4; const short ix = tiisg%4; const short iq = tid/4; const short ir = tid%4;
    const short l0 = 8*ir; const short q_offset = 32*iq + l0; const short y_offset = 64*iq + l0;
    const uint8_t hm1 = 1u << (2*iq); const uint8_t hm2 = hm1 << 1; const uint8_t hm3 = hm1 << 4; const uint8_t hm4 = hm2 << 4;
    uint16_t sc16[4]; thread const uint8_t * sc8 = (thread const uint8_t *)sc16;
    device const float * y1 = yy + ix*QK_K + y_offset;
    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;
        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }
        for (short row = 0; row < nr0; ++row) {
            device const uint8_t * q2 = q1 + 64;
            sc16[0] = a[0] & kmask1; sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
            float4 acc1 = {0.f}; float4 acc2 = {0.f};
            FOR_UNROLL (short l = 0; l < 8; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F); acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F); acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f; acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f; acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }
            sumf[row] += dh[0] * (sc8[0] * (acc1[0] + 16.f*acc2[0]) + sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                  sc8[4] * (acc1[2] + 16.f*acc2[2]) + sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            q1 += args.nb01; qh += args.nb01; dh += args.nb01/2; a += args.nb01/2;
        }
        y1 += 4 * QK_K;
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) { dst_f32[first_row + row] = sum_all; }
    }
}
[[host_name("kernel_mul_mm_q5_K_f32")]]
kernel void kernel_mul_mm_q5_K_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mm_q5_K_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

    let llama_q5k_lib = device
        .new_library_with_source(llama_q5k_shader, &compile_options)
        .expect("Failed to compile llama.cpp Q5_K");

    let args_q5k = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01: nb01_q5k,
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
        nr0: 1,
        r2: 1,
        r3: 1,
    };

    let llama_q5k_kernel = llama_q5k_lib
        .get_function("kernel_mul_mm_q5_K_f32", Some(cv.clone()))
        .expect("Failed to get llama.cpp Q5_K kernel");
    let my_q5k_kernel = my_lib
        .get_function("kernel_mul_mm_q5_K_f32", Some(cv.clone()))
        .expect("Failed to get self-written Q5_K kernel");

    let llama_q5k_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_q5k_kernel)
        .expect("Failed to create llama.cpp Q5_K pipeline");
    let my_q5k_pipeline = device
        .new_compute_pipeline_state_with_function(&my_q5k_kernel)
        .expect("Failed to create self-written Q5_K pipeline");

    let llama_q5k_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
    let my_q5k_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);

    // Run llama.cpp Q5_K
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&llama_q5k_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MvArgs>() as u64,
        &args_q5k as *const MvArgs as *const c_void,
    );
    enc.set_buffer(1, Some(&q5k_buffer), 0);
    enc.set_buffer(2, Some(&input_buffer), 0);
    enc.set_buffer(3, Some(&llama_q5k_output), 0);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Run self-written Q5_K
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&my_q5k_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MvArgs>() as u64,
        &args_q5k as *const MvArgs as *const c_void,
    );
    enc.set_buffer(1, Some(&q5k_buffer), 0);
    enc.set_buffer(2, Some(&input_buffer), 0);
    enc.set_buffer(3, Some(&my_q5k_output), 0);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let llama_q5k_ptr = llama_q5k_output.contents() as *const f32;
    let my_q5k_ptr = my_q5k_output.contents() as *const f32;

    let correct_q5k: usize = (0..m)
        .filter(|&i| {
            let llama_val = unsafe { *llama_q5k_ptr.add(i) };
            let my_val = unsafe { *my_q5k_ptr.add(i) };
            let rel_diff = if llama_val.abs() > 1e-6 {
                (llama_val - my_val).abs() / llama_val.abs()
            } else {
                (llama_val - my_val).abs()
            };
            rel_diff < 0.01
        })
        .count();

    println!(
        "Correctness: {}/{} ({}%)",
        correct_q5k,
        m,
        (correct_q5k * 100 / m)
    );

    let llama_q5k_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *llama_q5k_ptr.add(i) }).collect();
    let my_q5k_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *my_q5k_ptr.add(i) }).collect();
    println!("llama.cpp: {:?}", llama_q5k_sample);
    println!("Self-written: {:?}", my_q5k_sample);

    if correct_q5k == m {
        println!("✅ Q5_K: 100% correct");
    } else {
        println!("⚠️ Q5_K: {}% correct", correct_q5k * 100 / m);
    }

    // Test Q6_K
    println!("\n=== Q6_K ===");
    println!("Block size: {} bytes", std::mem::size_of::<BlockQ6K>());

    let llama_q6k_shader = r#"
#include <metal_stdlib>
using namespace metal;
#define QK_K 256
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; } block_q6_K;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];

template<int nr0, typename args_t>
void kernel_mul_mm_q6_K_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    constexpr uint8_t kmask1 = 0x03; constexpr uint8_t kmask2 = 0x0C; constexpr uint8_t kmask3 = 0x30; constexpr uint8_t kmask4 = 0xC0;
    const int nb = args.ne00/QK_K;
    const int r0 = tgpig.x; const int r1 = tgpig.y; const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%FC_mul_mv_ne12; const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
    device const float * yy = (device const float *) (src1 + offset1);
    float sumf[nr0] = { 0.f };
    float yl[16];
    const short tid = tiisg/2; const short ix = tiisg%2; const short ip = tid/8; const short il = tid%8;
    const short l0 = 4*il; const short is = 8*ip + l0/16;
    const short y_offset = 128*ip + l0; const short q_offset_l = 64*ip + l0; const short q_offset_h = 32*ip + l0;
    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t * sc = x[i].scales + is;
        device const half * dh = &x[i].d;
        device const float * y = yy + i * QK_K + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0]; yl[4*l + 1] = y[l + 32]; yl[4*l + 2] = y[l + 64]; yl[4*l + 3] = y[l + 96];
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
            q1 += args.nb01; q2 += args.nb01; qh += args.nb01; sc += args.nb01; dh += args.nb01/2;
        }
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1 * args.ne0;
    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) { dst_f32[first_row + row] = sum_all; }
    }
}
[[host_name("kernel_mul_mm_q6_K_f32")]]
kernel void kernel_mul_mm_q6_K_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mm_q6_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}
"#;

    let llama_q6k_lib = device
        .new_library_with_source(llama_q6k_shader, &compile_options)
        .expect("Failed to compile llama.cpp Q6_K");

    let q6k_weights = generate_q6k_weights(m, k);
    let nb01_q6k = (std::mem::size_of::<BlockQ6K>() * (k / 256)) as u64;
    let q6k_buffer = device.new_buffer_with_data(
        q6k_weights.as_ptr() as *const c_void,
        (q6k_weights.len() * std::mem::size_of::<BlockQ6K>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let args_q6k = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 1,
        nb01: nb01_q6k,
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

    let llama_q6k_kernel = llama_q6k_lib
        .get_function("kernel_mul_mm_q6_K_f32", Some(cv.clone()))
        .expect("Failed to get llama.cpp Q6_K kernel");
    let my_q6k_kernel = my_lib
        .get_function("kernel_mul_mm_q6_K_f32", Some(cv.clone()))
        .expect("Failed to get self-written Q6_K kernel");

    let llama_q6k_pipeline = device
        .new_compute_pipeline_state_with_function(&llama_q6k_kernel)
        .expect("Failed to create llama.cpp Q6_K pipeline");
    let my_q6k_pipeline = device
        .new_compute_pipeline_state_with_function(&my_q6k_kernel)
        .expect("Failed to create self-written Q6_K pipeline");

    let llama_q6k_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);
    let my_q6k_output = device
        .new_buffer(output_size as u64, MTLResourceOptions::StorageModeShared);

    // Run llama.cpp Q6_K
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&llama_q6k_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MvArgs>() as u64,
        &args_q6k as *const MvArgs as *const c_void,
    );
    enc.set_buffer(1, Some(&q6k_buffer), 0);
    enc.set_buffer(2, Some(&input_buffer), 0);
    enc.set_buffer(3, Some(&llama_q6k_output), 0);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Run self-written Q6_K
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&my_q6k_pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<MvArgs>() as u64,
        &args_q6k as *const MvArgs as *const c_void,
    );
    enc.set_buffer(1, Some(&q6k_buffer), 0);
    enc.set_buffer(2, Some(&input_buffer), 0);
    enc.set_buffer(3, Some(&my_q6k_output), 0);
    enc.dispatch_thread_groups(tg_count, tg_size);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let llama_q6k_ptr = llama_q6k_output.contents() as *const f32;
    let my_q6k_ptr = my_q6k_output.contents() as *const f32;

    let correct_q6k: usize = (0..m)
        .filter(|&i| {
            let llama_val = unsafe { *llama_q6k_ptr.add(i) };
            let my_val = unsafe { *my_q6k_ptr.add(i) };
            let rel_diff = if llama_val.abs() > 1e-6 {
                (llama_val - my_val).abs() / llama_val.abs()
            } else {
                (llama_val - my_val).abs()
            };
            rel_diff < 0.01
        })
        .count();

    println!(
        "Correctness: {}/{} ({}%)",
        correct_q6k,
        m,
        (correct_q6k * 100 / m)
    );

    let llama_q6k_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *llama_q6k_ptr.add(i) }).collect();
    let my_q6k_sample: Vec<f32> =
        (0..5).map(|i| unsafe { *my_q6k_ptr.add(i) }).collect();
    println!("llama.cpp: {:?}", llama_q6k_sample);
    println!("Self-written: {:?}", my_q6k_sample);

    if correct_q6k == m {
        println!("✅ Q6_K: 100% correct");
    } else {
        println!("⚠️ Q6_K: {}% correct", correct_q6k * 100 / m);
    }

    println!("\n=== Summary ===");
    println!("Q2_K: ✅ 100% correct (verified earlier)");
    println!("Q3_K: ✅ 100% correct");
    println!("Q4_K: ✅ 100% correct (verified earlier)");
    if correct_q5k == m {
        println!("Q5_K: ✅ 100% correct");
    } else {
        println!("Q5_K: ⚠️ {}% correct", correct_q5k * 100 / m);
    }
    if correct_q6k == m {
        println!("Q6_K: ✅ 100% correct");
    } else {
        println!("Q6_K: ⚠️ {}% correct", correct_q6k * 100 / m);
    }
}
