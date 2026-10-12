// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Direct comparison: Our Flash Decode vs llama.cpp for small seq_k

use infer_train::quant::kv_cache::{
    quantize_row_q4_0, quantize_row_q4_1, quantize_row_q5_0, quantize_row_q5_1,
    quantize_row_q8_0, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0,
};
use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 500;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    // Load our kernels
    let our_source =
        include_str!("../src/ops/metal/attention/flash_decode_multi_wg.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library =
        device.new_library_with_source(our_source, &options).unwrap();

    // Load llama.cpp kernels - we'll use a minimal subset
    let llama_source = create_minimal_llama_kernel();
    let llama_library =
        device.new_library_with_source(&llama_source, &options).unwrap();

    println!("=== Small seq_k: Our Flash Decode vs llama.cpp ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds\n",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );

    // Test small seq_k for F16 (most representative format)
    let seq_ks = [1, 8, 16, 32, 64, 128, 256];

    println!("\n=== F16 Format (baseline comparison) ===");
    println!(
        "{:>8} {:>12} {:>12} {:>10} {:>10} {:>8}",
        "seq_k", "Ours (µs)", "llama (µs)", "Ours GF", "llama GF", "Speedup"
    );
    println!("{}", "-".repeat(70));

    for &seq_k in &seq_ks {
        let result =
            benchmark_f16(&device, &queue, &our_library, &llama_library, seq_k);
        let speedup = if result.llama_time_us > 0.0 {
            result.llama_time_us / result.our_time_us
        } else {
            1.0
        };
        println!(
            "{:>8} {:>12.2} {:>12.2} {:>10.2} {:>10.2} {:>7.2}x",
            seq_k,
            result.our_time_us,
            result.llama_time_us,
            result.our_gflops,
            result.llama_gflops,
            speedup
        );
    }

    println!("\n=== Our kernel: All formats ===");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "seq_k",
        "F16 GF",
        "Q8_0 GB",
        "Q5_0 GF",
        "Q5_1 GF",
        "Q4_0 GF",
        "Q4_1 GF"
    );
    println!("{}", "-".repeat(80));

    for &seq_k in &seq_ks {
        let f16 = bench_our_only(&device, &queue, &our_library, "f16", seq_k);
        let q8_0 = bench_our_only(&device, &queue, &our_library, "q8_0", seq_k);
        let q5_0 = bench_our_only(&device, &queue, &our_library, "q5_0", seq_k);
        let q5_1 = bench_our_only(&device, &queue, &our_library, "q5_1", seq_k);
        let q4_0 = bench_our_only(&device, &queue, &our_library, "q4_0", seq_k);
        let q4_1 = bench_our_only(&device, &queue, &our_library, "q4_1", seq_k);
        println!(
            "{:>8} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            seq_k, f16, q8_0, q5_0, q5_1, q4_0, q4_1
        );
    }

    println!("\n=== Overhead Analysis ===");
    println!("Fixed kernel time at seq_k=1 represents launch + sync overhead");
}

struct BenchmarkResult {
    our_time_us: f64,
    llama_time_us: f64,
    our_gflops: f64,
    llama_gflops: f64,
}

fn benchmark_f16(
    device: &Device,
    queue: &CommandQueue,
    our_library: &Library,
    llama_library: &Library,
    seq_k: usize,
) -> BenchmarkResult {
    // Create F16 data
    let q_data: Vec<u16> = vec![0x3C00; NUM_HEADS * HEAD_DIM];
    let k_data: Vec<u16> = vec![0x3800; seq_k * NUM_HEADS * HEAD_DIM];
    let v_data: Vec<u16> = vec![0x3400; seq_k * NUM_HEADS * HEAD_DIM];

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

    // Our kernel setup
    let nwg = 8;
    let our_out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4 * nwg) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let partial_buf = device.new_buffer(
        (NUM_HEADS * 16 * nwg) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
    let scale_buf = device.new_buffer_with_data(
        &scale as *const f32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let seq_k_buf = device.new_buffer_with_data(
        &(seq_k as u32) as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let our_kernel_name =
        format!("flash_decode_multi_wg_partial_f16_c64_nwg{}", nwg);
    let our_func = our_library.get_function(&our_kernel_name, None).unwrap();
    let our_pipeline =
        device.new_compute_pipeline_state_with_function(&our_func).unwrap();

    let finalize_func = our_library
        .get_function(
            &format!("flash_decode_multi_wg_finalize_f16_nwg{}", nwg),
            None,
        )
        .unwrap();
    let finalize_pipeline = device
        .new_compute_pipeline_state_with_function(&finalize_func)
        .unwrap();

    // llama.cpp kernel setup
    let llama_out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let llama_func =
        llama_library.get_function("llama_flash_attn_f16", None).unwrap();
    let llama_pipeline =
        device.new_compute_pipeline_state_with_function(&llama_func).unwrap();

    // Warmup
    for _ in 0..50 {
        run_our_kernel(
            queue,
            &our_pipeline,
            &finalize_pipeline,
            &q_buf,
            &k_buf,
            &v_buf,
            &our_out_buf,
            &partial_buf,
            &scale_buf,
            &seq_k_buf,
            nwg,
        );
        run_llama_kernel(
            &device,
            queue,
            &llama_pipeline,
            &q_buf,
            &k_buf,
            &v_buf,
            &llama_out_buf,
            seq_k,
        );
    }

    // Alternating benchmark
    let mut our_times = Vec::new();
    let mut llama_times = Vec::new();

    for _ in 0..NUM_ROUNDS {
        let start = Instant::now();
        run_our_kernel(
            queue,
            &our_pipeline,
            &finalize_pipeline,
            &q_buf,
            &k_buf,
            &v_buf,
            &our_out_buf,
            &partial_buf,
            &scale_buf,
            &seq_k_buf,
            nwg,
        );
        our_times.push(start.elapsed().as_nanos() as f64);

        let start = Instant::now();
        run_llama_kernel(
            &device,
            queue,
            &llama_pipeline,
            &q_buf,
            &k_buf,
            &v_buf,
            &llama_out_buf,
            seq_k,
        );
        llama_times.push(start.elapsed().as_nanos() as f64);
    }

    let our_time_us =
        our_times.iter().sum::<f64>() / (NUM_ROUNDS as f64 * 1000.0);
    let llama_time_us =
        llama_times.iter().sum::<f64>() / (NUM_ROUNDS as f64 * 1000.0);

    let flops = 4.0 * seq_k as f64 * HEAD_DIM as f64 * NUM_HEADS as f64;
    let our_gflops = flops / our_time_us / 1000.0;
    let llama_gflops = flops / llama_time_us / 1000.0;

    BenchmarkResult { our_time_us, llama_time_us, our_gflops, llama_gflops }
}

fn bench_our_only(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
    seq_k: usize,
) -> f64 {
    let (k_buf, v_buf, _) = create_kv_data(device, format, seq_k);
    let q_buf = create_query_data(device);

    let nwg = 8;
    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4 * nwg) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let partial_buf = device.new_buffer(
        (NUM_HEADS * 16 * nwg) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
    let scale_buf = device.new_buffer_with_data(
        &scale as *const f32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let seq_k_buf = device.new_buffer_with_data(
        &(seq_k as u32) as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let kernel_name =
        format!("flash_decode_multi_wg_partial_{}_c64_nwg{}", format, nwg);
    let func = library.get_function(&kernel_name, None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&func).unwrap();

    let finalize_func = library
        .get_function(
            &format!("flash_decode_multi_wg_finalize_f16_nwg{}", nwg),
            None,
        )
        .unwrap();
    let finalize_pipeline = device
        .new_compute_pipeline_state_with_function(&finalize_func)
        .unwrap();

    // Warmup
    for _ in 0..50 {
        run_our_kernel(
            queue,
            &pipeline,
            &finalize_pipeline,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            &partial_buf,
            &scale_buf,
            &seq_k_buf,
            nwg,
        );
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..NUM_ROUNDS {
        run_our_kernel(
            queue,
            &pipeline,
            &finalize_pipeline,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            &partial_buf,
            &scale_buf,
            &seq_k_buf,
            nwg,
        );
    }
    let time_us =
        start.elapsed().as_nanos() as f64 / (NUM_ROUNDS as f64 * 1000.0);

    let flops = 4.0 * seq_k as f64 * HEAD_DIM as f64 * NUM_HEADS as f64;
    flops / time_us / 1000.0
}

fn create_query_data(device: &Device) -> Buffer {
    let q_data: Vec<u16> = vec![0x3C00; NUM_HEADS * HEAD_DIM];
    device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn create_kv_data(
    device: &Device,
    format: &str,
    seq_k: usize,
) -> (Buffer, Buffer, u32) {
    let num_elements = NUM_HEADS * seq_k * HEAD_DIM;
    let src_data: Vec<f32> = vec![0.5; num_elements];

    match format {
        "f16" => {
            let data: Vec<u16> = src_data
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            let size = (data.len() * 2) as u64;
            let buf = device.new_buffer_with_data(
                data.as_ptr() as *const _,
                size,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, (HEAD_DIM * 2) as u32)
        }
        "q8_0" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ8_0> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q8_0(&src_data, &mut blocks);
            let size = (num_blocks * 34) as u64;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, 34)
        }
        "q5_0" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ5_0> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q5_0(&src_data, &mut blocks);
            let size = (num_blocks * 22) as u64;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, 22)
        }
        "q5_1" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ5_1> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q5_1(&src_data, &mut blocks);
            let size = (num_blocks * 24) as u64;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, 24)
        }
        "q4_0" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ4_0> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q4_0(&src_data, &mut blocks);
            let size = (num_blocks * 18) as u64;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, 18)
        }
        "q4_1" => {
            let num_blocks = num_elements / 32;
            let mut blocks: Vec<BlockQ4_1> =
                vec![unsafe { std::mem::zeroed() }; num_blocks];
            quantize_row_q4_1(&src_data, &mut blocks);
            let size = (num_blocks * 20) as u64;
            let buf = device.new_buffer_with_data(
                blocks.as_ptr() as *const _,
                size,
                MTLResourceOptions::StorageModeShared,
            );
            (buf.clone(), buf, 20)
        }
        _ => panic!("Unknown format: {}", format),
    }
}

fn run_our_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    finalize_pipeline: &ComputePipelineState,
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    out_buf: &Buffer,
    partial_buf: &Buffer,
    scale_buf: &Buffer,
    seq_k_buf: &Buffer,
    nwg: usize,
) {
    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(seq_k_buf), 0);
    encoder.set_buffer(1, Some(scale_buf), 0);
    encoder.set_buffer(2, Some(q_buf), 0);
    encoder.set_buffer(3, Some(k_buf), 0);
    encoder.set_buffer(4, Some(v_buf), 0);
    encoder.set_buffer(5, Some(out_buf), 0);
    encoder.set_buffer(6, Some(partial_buf), 0);

    let threads_per_group = MTLSize::new(32, 1, 1);
    let num_groups = MTLSize::new(nwg as u64, NUM_HEADS as u64, 1);
    encoder.dispatch_thread_groups(num_groups, threads_per_group);
    encoder.end_encoding();

    let encoder2 = cmd_buf.new_compute_command_encoder();
    encoder2.set_compute_pipeline_state(finalize_pipeline);
    encoder2.set_buffer(0, Some(out_buf), 0);
    encoder2.set_buffer(1, Some(partial_buf), 0);
    encoder2.set_buffer(2, Some(out_buf), 0);
    encoder2.set_buffer(3, Some(seq_k_buf), 0);

    let threads_per_group2 = MTLSize::new(32, 1, 1);
    let num_groups2 = MTLSize::new(1, NUM_HEADS as u64, 1);
    encoder2.dispatch_thread_groups(num_groups2, threads_per_group2);
    encoder2.end_encoding();

    cmd_buf.commit();
    cmd_buf.wait_until_completed();
}

fn run_llama_kernel(
    device: &Device,
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    out_buf: &Buffer,
    seq_k: usize,
) {
    // Args struct matching llama.cpp's layout
    #[repr(C)]
    struct Args {
        ne11: u32, // seq_k
        nb11: u32, // K row stride
        nb21: u32, // V row stride
        scale: f32,
    }

    let args = Args {
        ne11: seq_k as u32,
        nb11: (HEAD_DIM * 2) as u32,
        nb21: (HEAD_DIM * 2) as u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
    };

    let args_buf = device.new_buffer_with_data(
        &args as *const Args as *const _,
        std::mem::size_of::<Args>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline);

    encoder.set_buffer(0, Some(&args_buf), 0);
    encoder.set_buffer(1, Some(q_buf), 0);
    encoder.set_buffer(2, Some(k_buf), 0);
    encoder.set_buffer(3, Some(v_buf), 0);
    encoder.set_buffer(4, Some(out_buf), 0);

    // Use 1 threadgroup per head, 32 threads
    let threads_per_group = MTLSize::new(32, 1, 1);
    let num_groups = MTLSize::new(NUM_HEADS as u64, 1, 1);
    encoder.dispatch_thread_groups(num_groups, threads_per_group);
    encoder.end_encoding();

    cmd_buf.commit();
    cmd_buf.wait_until_completed();
}

// Minimal llama.cpp flash attention kernel for testing
fn create_minimal_llama_kernel() -> String {
    r#"
#include <metal_stdlib>
using namespace metal;

// Minimal F16 flash attention for decode (seq_k < 256)
// Simplified baseline kernel - same algorithm as llama.cpp's fa_f16.metal

#define MAX_HALF 65504.0f

kernel void llama_flash_attn_f16(
    device const void* args_ptr [[buffer(0)]],
    device const half* query   [[buffer(1)]],
    device const half* key     [[buffer(2)]],
    device const half* value   [[buffer(3)]],
    device float* output       [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]]
) {
    // Args layout
    struct Args {
        uint ne11;   // seq_k
        uint nb11;   // K row stride (bytes)
        uint nb21;   // V row stride (bytes)
        float scale;
    };
    device const Args* args = (device const Args*)args_ptr;

    const uint head = tgpig.x;
    const uint seq_k = args->ne11;
    const float scale = args->scale;

    // Query for this head (1x128)
    device const half* q = query + head * 128;

    // K and V for this head
    device const half* k_base = key + head * 128;
    device const half* v_base = value + head * 128;

    // Initialize max and sum for online softmax
    float max_score = -MAX_HALF;
    float sum_exp = 0.0f;

    // Threadgroup memory for accumulation
    threadgroup float accum[128];
    for (int d = tiisg; d < 128; d += 32) {
        accum[d] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process each K position
    for (uint k_idx = 0; k_idx < seq_k; k_idx++) {
        // Q*K^T dot product (each thread computes partial sum)
        float qk = 0.0f;
        for (uint d = tiisg; d < 128; d += 32) {
            qk += (float)q[d] * (float)k_base[k_idx * 128 + d];
        }

        // Reduce across threads in simdgroup
        qk = simd_sum(qk);
        qk *= scale;

        // Online softmax update (only thread 0)
        if (tiisg == 0) {
            float new_max = max(max_score, qk);
            float exp_diff = exp(max_score - new_max);
            float exp_score = exp(qk - new_max);

            sum_exp = sum_exp * exp_diff + exp_score;
            max_score = new_max;

            // Update accumulation for V
            for (int d = 0; d < 128; d++) {
                accum[d] = accum[d] * exp_diff + exp_score * (float)v_base[k_idx * 128 + d];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Finalize output
    for (int d = tiisg; d < 128; d += 32) {
        output[head * 128 + d] = accum[d] / sum_exp;
    }
}
"#.to_string()
}
