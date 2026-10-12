// Verify MXFP4 MV kernel against llama.cpp Metal kernel
use metal::*;
use std::ffi::c_void;

#[repr(C)]
struct BlockMXFP4 {
    e: u8,
    qs: [u8; 16],
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

fn generate_mxfp4_weights(m: usize, k: usize) -> Vec<BlockMXFP4> {
    let nb = k / 32;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockMXFP4 {
                e: ((i % 256) as u8), // E8M0 scale
                qs,
            }
        })
        .collect()
}

fn generate_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

// llama.cpp MXFP4 kernel
const LLAMA_MXFP4_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define QK_MXFP4 32
#define N_SIMDWIDTH 32
#define FC_MUL_MV 600
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

typedef struct { uint8_t e; uint8_t qs[16]; } block_mxfp4;
typedef struct { int32_t ne00; int32_t ne01; int32_t ne02; uint64_t nb00; uint64_t nb01; uint64_t nb02; uint64_t nb03;
                 int32_t ne10; int32_t ne11; int32_t ne12; uint64_t nb10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
                 int32_t ne0; int32_t ne1; int32_t nr0; int16_t r2; int16_t r3; } ggml_metal_kargs_mul_mv;

constant short FC_mul_mv_nsg [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_ne12 [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2 [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3 [[function_constant(FC_MUL_MV + 4)]];

// e2m1 values (doubled), shared by MXFP4 and NVFP4
constexpr constant static float kvalues_mxfp4_f[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

static float e8m0_to_fp32(uint8_t e) {
    // E8M0 format: biased exponent with no mantissa
    // Value = 2^(e - 127) for e > 0, 0 for e = 0
    if (e == 0) return 0.0f;
    return exp2(float(e) - 127.0f);
}

template<int NR0, typename args_t>
void kernel_gemm_mxfp4_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, threadgroup char * shmem, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    threadgroup float * shmem_f32 = (threadgroup float *) shmem;
    const int r0 = tgpig.x; const int r1 = tgpig.y; const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * NR0;
    const uint i12 = im%FC_mul_mv_ne12; const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const block_mxfp4 * x = (device const block_mxfp4 *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);
    const int nb = args.ne00/QK_MXFP4;
    const int ns01 = args.nb01/args.nb00;
    const short ix = tiisg/2;
    const short it = tiisg%2;
    shmem_f32[tiisg] = kvalues_mxfp4_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 yl[4];
    float sumf[NR0]={0.f};
    device const float * yb = y + ix*QK_MXFP4 + it*8;
    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0]; yl[1] = y4[4]; yl[2] = y4[1]; yl[3] = y4[5];
        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_mxfp4 & xb = x[row*ns01 + ib];
            device const uint8_t * q2 = (device const uint8_t *)(xb.qs + 8*it);
            float4 acc1 = yl[0] * float4(shmem_f32[q2[0] & 0x0F], shmem_f32[q2[1] & 0x0F], shmem_f32[q2[2] & 0x0F], shmem_f32[q2[3] & 0x0F]);
            float4 acc2 = yl[1] * float4(shmem_f32[q2[0] >> 4], shmem_f32[q2[1] >> 4], shmem_f32[q2[2] >> 4], shmem_f32[q2[3] >> 4]);
            float4 acc3 = yl[2] * float4(shmem_f32[q2[4] & 0x0F], shmem_f32[q2[5] & 0x0F], shmem_f32[q2[6] & 0x0F], shmem_f32[q2[7] & 0x0F]);
            float4 acc4 = yl[3] * float4(shmem_f32[q2[4] >> 4], shmem_f32[q2[5] >> 4], shmem_f32[q2[6] >> 4], shmem_f32[q2[7] >> 4]);
            acc1 = (acc1 + acc3) + (acc2 + acc4);
            sumf[row] += e8m0_to_fp32(xb.e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
        }
        yb += 16 * QK_MXFP4;
    }
    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) { dst_f32[first_row + row] = sum_all; }
    }
}
[[host_name("kernel_gemm_mxfp4_f32")]]
kernel void kernel_gemm_mxfp4_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup char shmem[64];
    kernel_gemm_mxfp4_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}
"#;

fn main() {
    println!("=== MXFP4 MV Kernel Verification ===\n");

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let compile_options = CompileOptions::new();

    // Compile llama.cpp kernel
    let llama_lib = device
        .new_library_with_source(LLAMA_MXFP4_KERNEL, &compile_options)
        .expect("Failed to compile llama.cpp kernel");

    // Compile self-written template
    let shader_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let my_source = std::fs::read_to_string(shader_dir.join("gemm.metal"))
        .expect("Failed to read template");
    let my_lib = device
        .new_library_with_source(&my_source, &compile_options)
        .expect("Failed to compile template");

    let m = 256usize;
    let k = 256usize;
    let weights = generate_mxfp4_weights(m, k);
    let input = generate_input(k);

    let input_buffer = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        (k * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let weights_buffer = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        (weights.len() * 17) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cv = create_cv();
    let llama_kernel = llama_lib
        .get_function("kernel_gemm_mxfp4_f32", Some(cv.clone()))
        .unwrap();
    let my_kernel =
        my_lib.get_function("kernel_gemm_mxfp4_f32", Some(cv.clone())).unwrap();

    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_kernel).unwrap();
    let my_pipeline =
        device.new_compute_pipeline_state_with_function(&my_kernel).unwrap();

    let nb = k / 32;
    let nb01 = (17 * nb) as u64;
    let args = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: 17,
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
    enc.set_buffer(1, Some(&weights_buffer), 0);
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
    enc.set_buffer(1, Some(&weights_buffer), 0);
    enc.set_buffer(2, Some(&input_buffer), 0);
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
        println!("✅ MXFP4: 100% correct");
    } else {
        println!("⚠️ MXFP4: {}% correct", correct * 100 / m);
    }
}
