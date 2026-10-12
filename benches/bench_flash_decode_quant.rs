// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Alternating benchmark: Our Quantized Flash Decode vs llama.cpp

use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ROUNDS: usize = 200;

// FFI interface to llama.cpp
mod ffi {
    use std::os::raw::{c_char, c_float, c_void};

    #[repr(C)]
    pub struct FlashAttnResult {
        pub time_us: f64,
        pub status: i32,
        pub error_msg: [c_char; 256],
    }

    extern "C" {
        pub fn run_llama_flash_attn(
            format: *const c_char,
            q_data: *const c_float,
            k_data: *const c_void,
            v_data: *const c_void,
            out_data: *mut c_float,
            num_heads: i32,
            head_dim: i32,
            seq_k: i32,
            rounds: i32,
        ) -> FlashAttnResult;

        pub fn cleanup_llama_backend();
    }
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    // Load our kernels
    let our_source = include_str!("../src/ops/metal/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library =
        device.new_library_with_source(our_source, &options).unwrap();

    println!("=== Flash Decode Quantized: Ours vs llama.cpp ===");
    println!(
        "Config: {} heads, {} head_dim, {} rounds\n",
        NUM_HEADS, HEAD_DIM, NUM_ROUNDS
    );

    let seq_ks = [256, 512, 1024, 2048, 4096];
    let formats = ["q4_0", "q8_0"];

    for format in &formats {
        println!("\n=== Format: {} ===", format);
        println!(
            "{:>8} {:>12} {:>12} {:>8}",
            "seq_k", "Ours (µs)", "llama (µs)", "Ratio"
        );
        println!("{}", "-".repeat(48));

        for seq_k in &seq_ks {
            let result = benchmark_format_seq_k(
                &device,
                &queue,
                &our_library,
                format,
                *seq_k,
            );

            println!(
                "{:>8} {:>12.2} {:>12.2} {:>7.1}%",
                seq_k,
                result.our_time_us,
                result.llama_time_us,
                result.ratio_pct
            );
        }
    }

    // Cleanup
    unsafe {
        ffi::cleanup_llama_backend();
    }
}

struct BenchmarkResult {
    our_time_us: f64,
    llama_time_us: f64,
    ratio_pct: f64,
}

fn benchmark_format_seq_k(
    device: &Device,
    queue: &CommandQueue,
    our_library: &Library,
    format: &str,
    seq_k: usize,
) -> BenchmarkResult {
    // Create test data
    let q_data: Vec<f32> = vec![1.0; NUM_HEADS * HEAD_DIM];
    let k_f32: Vec<f32> = vec![1.0; seq_k * HEAD_DIM];
    let v_f32: Vec<f32> = vec![1.0; seq_k * HEAD_DIM];

    // Quantize K/V based on format
    let (k_quant, v_quant) =
        quantize_kv(format, &k_f32, &v_f32, seq_k, HEAD_DIM);

    // Create buffers for our kernel
    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_buf = device.new_buffer_with_data(
        k_quant.as_ptr() as *const _,
        k_quant.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer_with_data(
        v_quant.as_ptr() as *const _,
        v_quant.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // NSG for our kernel (adaptive based on seq_k)
    let nsg = compute_adaptive_nsg(seq_k);

    // Get our multi-group kernel
    let kernel_name = format!("flash_decode_multi_group_{}_g{}", format, nsg);
    let our_kernel = our_library.get_function(&kernel_name, None).unwrap();
    let our_pipeline =
        device.new_compute_pipeline_state_with_function(&our_kernel).unwrap();

    // Args for our kernel
    #[repr(C)]
    struct Args {
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

    let args = Args {
        batch: 1,
        heads: NUM_HEADS as u32,
        seq_q: 1,
        seq_k: seq_k as u32,
        head_dim: HEAD_DIM as u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        q_stride_b: (NUM_HEADS * HEAD_DIM * 4) as u32,
        q_stride_h: (HEAD_DIM * 4) as u32,
        q_stride_s: (HEAD_DIM * 4) as u32,
        k_stride_b: (seq_k * HEAD_DIM / 32 * get_block_size(format)) as u32,
        k_stride_h: (seq_k * HEAD_DIM / 32 * get_block_size(format)) as u32,
        k_stride_s: (HEAD_DIM / 32 * get_block_size(format)) as u32,
        v_stride_b: (seq_k * HEAD_DIM / 32 * get_block_size(format)) as u32,
        v_stride_h: (seq_k * HEAD_DIM / 32 * get_block_size(format)) as u32,
        v_stride_s: (HEAD_DIM / 32 * get_block_size(format)) as u32,
        o_stride_b: (NUM_HEADS * HEAD_DIM * 4) as u32,
        o_stride_h: (HEAD_DIM * 4) as u32,
        o_stride_s: (HEAD_DIM * 4) as u32,
    };

    let args_bytes = unsafe {
        std::slice::from_raw_parts(
            &args as *const _ as *const u8,
            std::mem::size_of::<Args>(),
        )
    };

    // Threadgroup memory
    let shmem_size = nsg * (2 + HEAD_DIM) * 4;

    // Warmup both
    for _ in 0..5 {
        run_our_kernel(
            &queue,
            &our_pipeline,
            args_bytes,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            NUM_HEADS,
            nsg,
            shmem_size,
        );
    }

    // llama.cpp warmup
    let mut llama_out = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    let format_cstr = std::ffi::CString::new(format).unwrap();
    unsafe {
        ffi::run_llama_flash_attn(
            format_cstr.as_ptr(),
            q_data.as_ptr(),
            k_quant.as_ptr() as *const _,
            v_quant.as_ptr() as *const _,
            llama_out.as_mut_ptr(),
            NUM_HEADS as i32,
            HEAD_DIM as i32,
            seq_k as i32,
            5,
        );
    }

    // Alternating benchmark
    let mut our_times = Vec::with_capacity(NUM_ROUNDS);
    let mut llama_times = Vec::with_capacity(NUM_ROUNDS);

    for _ in 0..NUM_ROUNDS {
        let t0 = Instant::now();
        run_our_kernel(
            &queue,
            &our_pipeline,
            args_bytes,
            &q_buf,
            &k_buf,
            &v_buf,
            &out_buf,
            NUM_HEADS,
            nsg,
            shmem_size,
        );
        our_times.push(t0.elapsed().as_nanos() as f64);

        let t1 = Instant::now();
        unsafe {
            ffi::run_llama_flash_attn(
                format_cstr.as_ptr(),
                q_data.as_ptr(),
                k_quant.as_ptr() as *const _,
                v_quant.as_ptr() as *const _,
                llama_out.as_mut_ptr(),
                NUM_HEADS as i32,
                HEAD_DIM as i32,
                seq_k as i32,
                1,
            );
        }
        llama_times.push(t1.elapsed().as_nanos() as f64);
    }

    our_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    llama_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let our_median = our_times[NUM_ROUNDS / 2] / 1000.0;
    let llama_median = llama_times[NUM_ROUNDS / 2] / 1000.0;
    let ratio = our_median / llama_median * 100.0;

    BenchmarkResult {
        our_time_us: our_median,
        llama_time_us: llama_median,
        ratio_pct: ratio,
    }
}

fn compute_adaptive_nsg(seq_k: usize) -> usize {
    if seq_k <= 128 {
        1
    } else if seq_k <= 256 {
        2
    } else if seq_k <= 512 {
        4
    } else if seq_k <= 1024 {
        4
    } else if seq_k <= 2048 {
        8
    } else {
        8
    }
}

fn run_our_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    args_bytes: &[u8],
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    out_buf: &Buffer,
    num_heads: usize,
    nsg: usize,
    shmem_size: usize,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(0, args_bytes.len() as u64, args_bytes.as_ptr() as *const _);
    enc.set_buffer(1, Some(q_buf), 0);
    enc.set_buffer(2, Some(k_buf), 0);
    enc.set_buffer(3, Some(v_buf), 0);
    enc.set_buffer(4, Some(out_buf), 0);
    if shmem_size > 0 {
        enc.set_threadgroup_memory_length(0, shmem_size as u64);
    }
    enc.dispatch_thread_groups(
        MTLSize::new(1, num_heads as u64, 1),
        MTLSize::new(32, nsg as u64, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn get_block_size(format: &str) -> usize {
    match format {
        "q4_0" => 18, // 2 bytes scale + 16 bytes qs (32 elements * 4 bits / 8)
        "q8_0" => 34, // 2 bytes scale + 32 bytes qs
        _ => 18,
    }
}

fn quantize_kv(
    format: &str,
    k_f32: &[f32],
    v_f32: &[f32],
    seq_k: usize,
    head_dim: usize,
) -> (Vec<u8>, Vec<u8>) {
    match format {
        "q4_0" => {
            use infer_train::quant::kv_cache::{quantize_row_q4_0, BlockQ4_0};
            let k = quantize_to_q4_0(k_f32, seq_k, head_dim);
            let v = quantize_to_q4_0(v_f32, seq_k, head_dim);
            (k, v)
        }
        "q8_0" => {
            use infer_train::quant::kv_cache::{quantize_row_q8_0, BlockQ8_0};
            let k = quantize_to_q8_0(k_f32, seq_k, head_dim);
            let v = quantize_to_q8_0(v_f32, seq_k, head_dim);
            (k, v)
        }
        _ => panic!("Unknown format: {}", format),
    }
}

fn quantize_to_q4_0(data: &[f32], seq: usize, head_dim: usize) -> Vec<u8> {
    use infer_train::quant::kv_cache::{quantize_row_q4_0, BlockQ4_0};
    let blocks_per_row = head_dim / 32;
    let mut result = Vec::with_capacity(
        seq * blocks_per_row * std::mem::size_of::<BlockQ4_0>(),
    );

    for s in 0..seq {
        let row_data = &data[s * head_dim..(s + 1) * head_dim];
        let mut blocks: Vec<BlockQ4_0> =
            vec![unsafe { std::mem::zeroed() }; blocks_per_row];
        quantize_row_q4_0(row_data, &mut blocks);
        let block_bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                blocks.len() * std::mem::size_of::<BlockQ4_0>(),
            )
        };
        result.extend_from_slice(block_bytes);
    }
    result
}

fn quantize_to_q8_0(data: &[f32], seq: usize, head_dim: usize) -> Vec<u8> {
    use infer_train::quant::kv_cache::{quantize_row_q8_0, BlockQ8_0};
    let blocks_per_row = head_dim / 32;
    let mut result = Vec::with_capacity(
        seq * blocks_per_row * std::mem::size_of::<BlockQ8_0>(),
    );

    for s in 0..seq {
        let row_data = &data[s * head_dim..(s + 1) * head_dim];
        let mut blocks: Vec<BlockQ8_0> =
            vec![unsafe { std::mem::zeroed() }; blocks_per_row];
        quantize_row_q8_0(row_data, &mut blocks);
        let block_bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                blocks.len() * std::mem::size_of::<BlockQ8_0>(),
            )
        };
        result.extend_from_slice(block_bytes);
    }
    result
}
