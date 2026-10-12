// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Compare our MPP prefill with llama.cpp's tensor kernel

use metal::*;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);

    // Load our MPP kernel
    let our_source = include_str!("../../shaders/flash_prefill_mpp.metal");
    let our_source_def =
        format!("#define GGML_METAL_HAS_TENSOR\n{}", our_source);

    println!("Compiling our kernel...");
    let our_lib = device
        .new_library_with_source(&our_source_def, &options)
        .expect("Failed to compile our kernel");
    let our_func = our_lib
        .get_function("flash_prefill_mpp_f16_d128", None)
        .expect("Kernel not found");
    let our_pipeline =
        device.new_compute_pipeline_state_with_function(&our_func).unwrap();
    println!("Our kernel compiled successfully");

    // Load llama.cpp tensor kernel from fa_f16.metal
    // We need to provide all the necessary defines and inline the dependencies
    let llama_source = r#"
#define GGML_METAL_HAS_TENSOR

#include <metal_stdlib>
using namespace metal;

#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Minimal definitions from llama.cpp
#define N_SIMDWIDTH 32
#define OP_FLASH_ATTN_EXT_TENSOR_NQPSG 8
#define OP_FLASH_ATTN_EXT_TENSOR_NCPSG 256
#define OP_FLASH_ATTN_EXT_TENSOR_NSG   4
#define OP_FLASH_ATTN_EXT_TENSOR       700

constant bool FC_flash_attn_ext_tensor_has_mask  [[function_constant(OP_FLASH_ATTN_EXT_TENSOR + 0)]];
constant bool FC_flash_attn_ext_tensor_has_sinks [[function_constant(OP_FLASH_ATTN_EXT_TENSOR + 1)]];
constant bool FC_flash_attn_ext_tensor_has_bias  [[function_constant(OP_FLASH_ATTN_EXT_TENSOR + 2)]];
constant bool FC_flash_attn_ext_tensor_has_scap  [[function_constant(OP_FLASH_ATTN_EXT_TENSOR + 3)]];

// Args structure (simplified)
struct ggml_metal_kargs_flash_attn_ext {
    uint32_t ne01;    // Q rows
    uint32_t ne02;    // Q heads
    uint32_t ne03;    // Q batches
    uint32_t ne11;    // K rows (seq_len)
    uint32_t ne_12_2; // KV heads
    uint32_t ne_12_3; // KV batches
    uint32_t nb01;    // Q stride (bytes)
    uint32_t nb02;    // Q stride (bytes)
    uint32_t nb03;    // Q stride (bytes)
    uint32_t nb11;    // K stride (bytes)
    uint32_t nb12;    // K stride (bytes)
    uint32_t nb13;    // K stride (bytes)
    uint32_t nb21;    // V stride (bytes)
    uint32_t nb22;    // V stride (bytes)
    uint32_t nb23;    // V stride (bytes)
    uint32_t nb31;    // mask stride
    uint32_t nb32;    // mask stride
    uint32_t nb33;    // mask stride
    uint32_t ne32;    // mask rows
    uint32_t ne33;    // mask cols
    float scale;
    float max_bias;
    float logit_softcap;
    uint32_t n_head_log2;
    float m0;
    float m1;
    uint32_t ns10;  // K stride in elements
    uint32_t ns20;  // V stride in elements
};

#define PAD2(x, a) (((x) + (a) - 1) & ~((a) - 1))
#define MAXHALF 65504.0h

template<short Q, short C, short NSG, short DK, short DV>
kernel void kernel_flash_attn_ext_tensor(
    constant ggml_metal_kargs_flash_attn_ext & args,
    device const char * q,
    device const char * k,
    device const char * v,
    device const char * mask,
    device const char * sinks,
    device const char * blk,
    device char * dst,
    threadgroup char * shmem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NT = NW*NSG;
    constexpr short NQ = Q/NSG;
    constexpr short NC = C/NW;

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0]*Q;

    const short tiitg = sgitg*NW + tiisg;

    threadgroup half  * sq = (threadgroup half  *) shmem;
    threadgroup float * ss = (threadgroup float *) (sq + Q*DK);
    threadgroup half  * sp = (threadgroup half  *) (ss + Q*C);
    threadgroup float * sr = (threadgroup float *) (sp + Q*C);
    threadgroup int   * sf = (threadgroup int   *) (sr + Q);

    q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

    {
        const int ikv2 = iq2/(args.ne02/args.ne_12_2);
        const int ikv3 = iq3/(args.ne03/args.ne_12_3);
        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    const float qscale = FC_flash_attn_ext_tensor_has_scap ? 1.0f : args.scale;

    // Load queries
    for (int i = tiitg; i < Q*DK/4; i += NT) {
        const int j = i/(DK/4);
        float4 q4 = 0.0f;
        if (iq1 + j < args.ne01) {
            q4 = ((device const float4 *) (q + j*args.nb01))[i%(DK/4)];
        }
        ((threadgroup half4 *) sq)[i] = (half4)(q4*qscale);
    }

    float M[NQ];
    float S[NQ];

    for (short jj = 0; jj < NQ; jj++) {
        M[jj] = -FLT_MAX/2;
        S[jj] = 0.0f;
    }

    const int sk = args.ns10;
    const int sv = args.ns20;

    auto tq = tensor(sq, dextents<int32_t, 2>(DK, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C, Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C, Q));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, DV, C, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    auto tv0 = tensor((device half *) v, dextents<int32_t, 2>(DV, args.ne11), array<int, 2>({1, sv}));
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();

    for (short i = 0; i < co.get_capacity(); i++) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    if (tiitg == 0) {
        sf[0] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main loop
    for (int ic0 = 0, ic = 0; ic < args.ne11; ic0++, ic += C) {
        // Q*K^T
        {
            auto tk = tensor((device half *) (k + (uint64_t) ic*args.nb11), dextents<int32_t, 2>(DK, C), array<int, 2>({1, sk}));
            mm_qk.run(tq, tk, ts);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax
        for (short jj = 0; jj < NQ; jj++) {
            const short j = jj*NSG + sgitg;

            if (iq1 + j >= args.ne01) continue;

            float s[NC];
            for (short ii = 0; ii < NC; ii++) {
                s[ii] = ss[j*C + ii*NW + tiisg];
            }

            float m = M[jj];
            for (short ii = 0; ii < NC; ii++) {
                m = max(m, s[ii]);
            }
            m = simd_max(m);

            float ms = 1.0f;
            if (m > M[jj] + 8.0f) {
                ms = exp(M[jj] - m);
                M[jj] = m;
                if (tiisg == 0) {
                    sf[0] = ic0 + 1;
                }
            }

            float sum = 0.0f;
            for (short ii = 0; ii < NC; ii++) {
                const half p = (half) exp(s[ii] - M[jj]);
                sp[j*C + ii*NW + tiisg] = p;
                sum += (float) p;
            }

            S[jj] = S[jj]*ms + simd_sum(sum);

            if (tiisg == 0) {
                sr[j] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // P*V
        if (sf[0] == ic0 + 1) {
            for (short i = 0; i < co.get_capacity(); i++) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        {
            auto tv = tensor((device half *) (v + (uint64_t) ic*args.nb21), dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));
            mm_pv.run(tp, tv, co);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    device half * dst_half = (device half *) dst;
    for (short jj = 0; jj < NQ; jj++) {
        const short j = jj*NSG + sgitg;
        if (iq1 + j >= args.ne01) continue;

        for (short d = tiisg; d < DV; d += NW) {
            float val = co[d + j*DV] / S[jj];
            dst_half[(iq1 + j)*args.nb01/2 + d + iq2*args.nb02/2 + iq3*args.nb03/2] = (half) val;
        }
    }
}

template [[host_name("kernel_flash_attn_ext_tensor_f16_dk128_dv128")]]
kernel void kernel_flash_attn_ext_tensor<8, 256, 4, 128, 128>(
    constant ggml_metal_kargs_flash_attn_ext &, device const char *, device const char *, device const char *,
    device const char *, device const char *, device const char *, device char *,
    threadgroup char *, uint3, ushort, ushort);
"#;

    println!("Compiling llama.cpp kernel...");
    let llama_lib = match device.new_library_with_source(llama_source, &options)
    {
        Ok(lib) => {
            println!("llama.cpp kernel compiled successfully");
            lib
        }
        Err(e) => {
            eprintln!("Failed to compile llama.cpp kernel: {}", e);
            run_our_only(&device, &queue, &our_pipeline);
            return;
        }
    };

    let llama_func = match llama_lib
        .get_function("kernel_flash_attn_ext_tensor_f16_dk128_dv128", None)
    {
        Ok(f) => f,
        Err(_) => {
            eprintln!("llama.cpp kernel not found");
            run_our_only(&device, &queue, &our_pipeline);
            return;
        }
    };
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_func).unwrap();

    run_comparison(&device, &queue, &our_pipeline, &llama_pipeline);
}

fn run_our_only(
    device: &Device,
    queue: &CommandQueue,
    our_pipeline: &ComputePipelineState,
) {
    println!("\n=== Our MPP Prefill Performance ===");
    println!("seq_q  seq_k   Time(µs)  GFLOPS");

    let test_cases = [(16, 256), (64, 512), (128, 1024), (256, 2048)];

    for (seq_q, seq_k) in test_cases {
        let (time_us, gflops) =
            run_our_prefill(device, queue, our_pipeline, seq_q, seq_k);
        println!("{:4}  {:5}  {:8.2}  {:7.2}", seq_q, seq_k, time_us, gflops);
    }
}

fn run_comparison(
    device: &Device,
    queue: &CommandQueue,
    our_pipeline: &ComputePipelineState,
    llama_pipeline: &ComputePipelineState,
) {
    println!("\n=== Prefill Performance Comparison ===");
    println!("seq_q  seq_k   Ours(GF) llama(GF)   Ratio   Status");

    let test_cases = [(16, 256), (64, 512), (128, 1024), (256, 2048)];

    for (seq_q, seq_k) in test_cases {
        let (our_time, our_gflops) =
            run_our_prefill(device, queue, our_pipeline, seq_q, seq_k);
        let (_llama_time, llama_gflops) =
            run_llama_prefill(device, queue, llama_pipeline, seq_q, seq_k);

        let ratio = our_gflops / llama_gflops * 100.0;
        let status = if ratio >= 100.0 { "✅" } else { "❌" };

        println!(
            "{:4}  {:5}  {:8.1}  {:9.1}  {:6.1}%  {}",
            seq_q, seq_k, our_gflops, llama_gflops, ratio, status
        );
    }
}

fn run_our_prefill(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    seq_q: usize,
    seq_k: usize,
) -> (f64, f64) {
    // Our args structure
    #[repr(C)]
    struct OurArgs {
        batch: u32,
        heads: u32,
        seq_q: u32,
        seq_k: u32,
        head_dim: u32,
        scale: f32,
        q_stride_b: u32,
        q_stride_h: u32,
        q_stride_s: u32,
        k_stride_b: u32,
        k_stride_h: u32,
        k_stride_s: u32,
        v_stride_b: u32,
        v_stride_h: u32,
        v_stride_s: u32,
        o_stride_b: u32,
        o_stride_h: u32,
        o_stride_s: u32,
    }

    run_bench(
        device,
        queue,
        pipeline,
        seq_q,
        seq_k,
        |args| {
            let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
            OurArgs {
                batch: 1,
                heads: NUM_HEADS as u32,
                seq_q: seq_q as u32,
                seq_k: seq_k as u32,
                head_dim: HEAD_DIM as u32,
                scale,
                q_stride_b: (NUM_HEADS * HEAD_DIM) as u32,
                q_stride_h: HEAD_DIM as u32,
                q_stride_s: HEAD_DIM as u32,
                k_stride_b: (NUM_HEADS * HEAD_DIM) as u32,
                k_stride_h: HEAD_DIM as u32,
                k_stride_s: HEAD_DIM as u32,
                v_stride_b: (NUM_HEADS * HEAD_DIM) as u32,
                v_stride_h: HEAD_DIM as u32,
                v_stride_s: HEAD_DIM as u32,
                o_stride_b: (NUM_HEADS * HEAD_DIM) as u32,
                o_stride_h: HEAD_DIM as u32,
                o_stride_s: HEAD_DIM as u32,
            }
        },
        std::mem::size_of::<OurArgs>(),
    )
}

fn run_llama_prefill(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    seq_q: usize,
    seq_k: usize,
) -> (f64, f64) {
    // llama.cpp args structure
    #[repr(C)]
    struct LlamaArgs {
        ne01: u32,
        ne02: u32,
        ne03: u32,
        ne11: u32,
        ne_12_2: u32,
        ne_12_3: u32,
        nb01: u32,
        nb02: u32,
        nb03: u32,
        nb11: u32,
        nb12: u32,
        nb13: u32,
        nb21: u32,
        nb22: u32,
        nb23: u32,
        nb31: u32,
        nb32: u32,
        nb33: u32,
        ne32: u32,
        ne33: u32,
        scale: f32,
        max_bias: f32,
        logit_softcap: f32,
        n_head_log2: u32,
        m0: f32,
        m1: f32,
        ns10: u32,
        ns20: u32,
    }

    run_bench(
        device,
        queue,
        pipeline,
        seq_q,
        seq_k,
        |_args| {
            let stride_b = (NUM_HEADS * HEAD_DIM * 2) as u32;
            let stride_s = (HEAD_DIM * 2) as u32;
            LlamaArgs {
                ne01: seq_q as u32,
                ne02: NUM_HEADS as u32,
                ne03: 1,
                ne11: seq_k as u32,
                ne_12_2: NUM_HEADS as u32,
                ne_12_3: 1,
                nb01: stride_s,
                nb02: stride_b,
                nb03: stride_b,
                nb11: stride_s,
                nb12: stride_b,
                nb13: stride_b,
                nb21: stride_s,
                nb22: stride_b,
                nb23: stride_b,
                nb31: 0,
                nb32: 0,
                nb33: 0,
                ne32: 0,
                ne33: 0,
                scale: 1.0f32 / (HEAD_DIM as f32).sqrt(),
                max_bias: 0.0,
                logit_softcap: 0.0,
                n_head_log2: 0,
                m0: 0.0,
                m1: 0.0,
                ns10: HEAD_DIM as u32,
                ns20: HEAD_DIM as u32,
            }
        },
        std::mem::size_of::<LlamaArgs>(),
    )
}

fn run_bench<A, F>(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    seq_q: usize,
    seq_k: usize,
    make_args: F,
    args_size: usize,
) -> (f64, f64)
where
    F: FnOnce(&[u16]) -> A,
{
    // Create test data
    let q_data: Vec<u16> = vec![0x3C00; seq_q * NUM_HEADS * HEAD_DIM];
    let k_data: Vec<u16> = vec![0x3800; seq_k * NUM_HEADS * HEAD_DIM];
    let v_data: Vec<u16> = vec![0x3400; seq_k * NUM_HEADS * HEAD_DIM];
    let mask_data: Vec<u8> = vec![0; 16];
    let sink_data: Vec<f32> = vec![0.0; NUM_HEADS];

    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        k_data.as_ptr() as *const _,
        (k_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        v_data.as_ptr() as *const _,
        (v_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (seq_q * NUM_HEADS * HEAD_DIM * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let mask_buf = device.new_buffer_with_data(
        mask_data.as_ptr() as *const _,
        mask_data.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sink_buf = device.new_buffer_with_data(
        sink_data.as_ptr() as *const _,
        (sink_data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let args = make_args(&q_data);

    let threads_per_group = MTLSize::new(128, 1, 1);
    let q_per_tg = 8; // llama.cpp uses Q=8
    let num_groups = MTLSize::new(
        ((seq_q + q_per_tg - 1) / q_per_tg) as u64,
        NUM_HEADS as u64,
        1,
    );
    let shmem_size = 65536u64; // 64KB shared memory

    // Warmup
    {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(
            0,
            args_size as u64,
            &args as *const A as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&q_buf), 0);
        encoder.set_buffer(2, Some(&k_buf), 0);
        encoder.set_buffer(3, Some(&v_buf), 0);
        encoder.set_buffer(4, Some(&mask_buf), 0);
        encoder.set_buffer(5, Some(&sink_buf), 0);
        encoder.set_buffer(6, Some(&out_buf), 0);
        encoder.set_buffer(7, Some(&out_buf), 0);
        encoder.set_threadgroup_memory_length(0, shmem_size);
        encoder.dispatch_thread_groups(num_groups, threads_per_group);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
    }

    // Benchmark
    let iterations = 10;
    let mut total_time = 0.0f64;

    for _ in 0..iterations {
        let cmd_buf = queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(
            0,
            args_size as u64,
            &args as *const A as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&q_buf), 0);
        encoder.set_buffer(2, Some(&k_buf), 0);
        encoder.set_buffer(3, Some(&v_buf), 0);
        encoder.set_buffer(4, Some(&mask_buf), 0);
        encoder.set_buffer(5, Some(&sink_buf), 0);
        encoder.set_buffer(6, Some(&out_buf), 0);
        encoder.set_buffer(7, Some(&out_buf), 0);
        encoder.set_threadgroup_memory_length(0, shmem_size);
        encoder.dispatch_thread_groups(num_groups, threads_per_group);
        encoder.end_encoding();

        let start = std::time::Instant::now();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        total_time += start.elapsed().as_nanos() as f64 / 1000.0;
    }

    let time_us = total_time / iterations as f64;
    let flops =
        2.0 * seq_q as f64 * seq_k as f64 * HEAD_DIM as f64 * NUM_HEADS as f64;
    let gflops = flops / time_us / 1000.0;

    (time_us, gflops)
}
