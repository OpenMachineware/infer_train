// Alternating benchmark: Self-written vs llama.cpp Q4_K MV
// Minimal version: compile both kernels, measure alternating
use metal::*;
use std::ffi::c_void;
use std::time::Instant;

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

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(true);

    // Load grid tables
    let grid_tables = include_str!("../../shaders/iq_grid_tables.h");
    // Add missing tables for IQ kernels
    let iq_extra_tables = r#"
constexpr constant static uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
constexpr constant static int8_t ksigns_iq2xs[128] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,
     1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,
     1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,
     1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,
     1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
};
"#;

    // Compile llama.cpp kernel (extracted from mul_mv.metal, Q4_K only)
    let llama_q4k = r#"
#include <metal_stdlib>
using namespace metal;
#define QK_K 256
#define K_SCALE_SIZE 12
typedef struct { half d; half dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[128]; } block_q4_K;
constant short FC_nsg [[function_constant(600)]];
constant short FC_ne12 [[function_constant(602)]];
constant short FC_r2 [[function_constant(603)]];
constant short FC_r3 [[function_constant(604)]];

static inline float get_scale(int k, device const uint8_t * scales) {
    return ((scales[k] & 0xF) | ((scales[K_SCALE_SIZE + k/2] & 0xF) << 4)) - 32;
}

typedef struct {
    int32_t ne00,ne01,ne02,ne10,ne11,ne12,ne0,ne1,nr0;
    uint64_t nb00,nb01,nb02,nb03,nb10,nb11,nb12,nb13;
    int16_t r2,r3;
} ggml_metal_kargs_mul_mv;

template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(args_t args, device const char * src0, device const char * src1,
    device char * dst, uint3 tgpig, ushort tiisg, ushort sgitg) {
    const short NSG = FC_nsg;
    const int r0 = tgpig.x, r1 = tgpig.y, im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%FC_ne12, i13 = im/FC_ne12;
    const uint64_t off0 = first_row*args.nb01 + (i12/FC_r2)*args.nb02 + (i13/FC_r3)*args.nb03;
    const uint64_t off1 = r1*args.nb11 + i12*args.nb12 + i13*args.nb13;
    device const block_q4_K * x = (device const block_q4_K *)(src0 + off0);
    device const float * y = (device const float *)(src1 + off1);
    const int nb = args.ne00/QK_K, ns01 = args.nb01/args.nb00;
    float sumf[nr0] = {0.f};
    for (int ib = tiisg; ib < nb; ib += 32) {
        for (int row = 0; row < nr0; ++row) {
            device const block_q4_K * xb = x + row*ns01 + ib;
            float sumy = 0;
            for (int j = 0; j < 16; ++j) sumy += y[ib*QK_K + j] + y[ib*QK_K + j+16];
            const float d = xb->d * get_scale(ib%16, xb->scales);
            sumf[row] += d * sumy;
        }
    }
    device float * dst_f32 = (device float *)dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0 && first_row+row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) dst_f32[first_row+row] = sum_all;
    }
}
[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(constant ggml_metal_kargs_mul_mv & args, device const char * src0,
    device const char * src1, device char * dst, uint3 tgpig[[threadgroup_position_in_grid]],
    ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}
"#;

    let llama_lib = device
        .new_library_with_source(llama_q4k, &opts)
        .expect("Failed to compile llama kernel");

    // Compile our kernel (from gemm.metal)
    let our_source = include_str!("../../shaders/gemm.metal");
    let our_combined =
        format!("{}\n{}\n{}", iq_extra_tables, grid_tables, our_source);
    let our_lib = device
        .new_library_with_source(&our_combined, &opts)
        .expect("Failed to compile our kernel");

    let cv = create_cv();
    let llama_pipeline = device
        .new_compute_pipeline_state_with_function(
            &llama_lib
                .get_function("kernel_mul_mv_q4_K_f32", Some(cv.clone()))
                .unwrap(),
        )
        .unwrap();
    let our_pipeline = device
        .new_compute_pipeline_state_with_function(
            &our_lib.get_function("kernel_gemm_q4_K_f32", Some(cv)).unwrap(),
        )
        .unwrap();

    // Generate test data
    let m = 4096usize;
    let k = 4096usize;
    let seed = 42u64;
    let block_size = 144; // Q4_K
    let nb = k / 256;

    let mut weights = vec![0u8; m * nb * block_size];
    for i in 0..weights.len() {
        weights[i] = ((i * 17 + seed as usize) % 256) as u8;
    }

    let mut input = vec![0.0f32; k];
    for i in 0..k {
        input[i] = (i as f32 % 10.0) * 0.1;
    }

    let weights_buf = device.new_buffer_with_data(
        weights.as_ptr() as *const c_void,
        weights.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const c_void,
        input.len() as u64 * 4,
        MTLResourceOptions::StorageModeShared,
    );
    let llama_out = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);
    let our_out = device
        .new_buffer((m * 4) as u64, MTLResourceOptions::StorageModeShared);

    let args = MvArgs {
        ne00: k as i32,
        ne01: m as i32,
        ne02: 1,
        nb00: block_size as u64,
        nb01: (nb * block_size) as u64,
        nb02: 0,
        nb03: 0,
        ne10: k as i32,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: (k * 4) as u64,
        nb12: 0,
        nb13: 0,
        ne0: m as i32,
        ne1: 1,
        nr0: 2,
        r2: 1,
        r3: 1,
    };

    let tg_size = MTLSize { width: 32, height: 2, depth: 1 };
    let tg_count = MTLSize { width: ((m + 1) / 2) as u64, height: 1, depth: 1 };

    // Warmup
    for _ in 0..50 {
        for pipeline in [&llama_pipeline, &our_pipeline] {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const _ as *const c_void,
            );
            enc.set_buffer(1, Some(&weights_buf), 0);
            enc.set_buffer(2, Some(&input_buf), 0);
            enc.set_buffer(3, Some(&llama_out), 0);
            enc.dispatch_thread_groups(tg_count, tg_size);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    // Alternating benchmark
    let rounds = 200;
    let mut llama_times = Vec::new();
    let mut our_times = Vec::new();

    for _ in 0..rounds {
        // llama.cpp
        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&llama_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&llama_out), 0);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        llama_times.push(start.elapsed().as_micros() as f64);

        // Ours
        let start = Instant::now();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&our_pipeline);
        enc.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const _ as *const c_void,
        );
        enc.set_buffer(1, Some(&weights_buf), 0);
        enc.set_buffer(2, Some(&input_buf), 0);
        enc.set_buffer(3, Some(&our_out), 0);
        enc.dispatch_thread_groups(tg_count, tg_size);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        our_times.push(start.elapsed().as_micros() as f64);
    }

    let avg_llama: f64 = llama_times.iter().sum::<f64>() / rounds as f64;
    let avg_ours: f64 = our_times.iter().sum::<f64>() / rounds as f64;
    let ratio = avg_ours / avg_llama * 100.0;

    println!(
        "=== Q4_K MV Alternating Benchmark (M={}, K={}, {} rounds) ===",
        m, k, rounds
    );
    println!("llama.cpp: {:.1} us", avg_llama);
    println!("Ours:      {:.1} us", avg_ours);
    println!(
        "Ratio:     {:.1}% ({})",
        ratio,
        if ratio < 100.0 {
            "FASTER"
        } else if ratio < 101.0 {
            "TIED"
        } else {
            "SLOWER"
        }
    );

    // Count wins
    let wins: usize =
        (0..rounds).filter(|&i| our_times[i] < llama_times[i]).count();
    println!("Wins:      {}/{}", wins, rounds);
}
