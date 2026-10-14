// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark GPU Flash Attention: Self-written vs llama.cpp
// Compares performance for formats both support

use infer_train::quant::kv_cache::{
    BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, quantize_row_q4_0,
    quantize_row_q4_1, quantize_row_q5_0, quantize_row_q5_1, quantize_row_q8_0,
};
use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const NUM_ITER: usize = 100;
const WARMUP: usize = 10;

fn main() {
    println!("=== GPU Flash Attention Benchmark ===");
    println!("Config: {} heads, {} head_dim\n", NUM_HEADS, HEAD_DIM);

    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    // Load our self-written kernels
    let our_source =
        include_str!("../src/ops/metal/attention/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library = device
        .new_library_with_source(our_source, &options)
        .expect("Failed to compile our library");

    // Try to load llama.cpp kernels
    let llama_source = match load_llama_cpp_kernels() {
        Ok(src) => Some(src),
        Err(e) => {
            println!("Warning: Could not load llama.cpp kernels: {}", e);
            println!("Running our implementation only.\n");
            None
        }
    };

    let llama_library = llama_source
        .as_ref()
        .and_then(|src| device.new_library_with_source(src, &options).ok());

    // Test formats
    let formats = ["f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"];

    println!("Decode (seq_q=1, seq_k=512):");
    println!(
        "{:>8} {:>15} {:>15} {:>10}",
        "Format", "Ours (GFLOPS)", "Time (µs)", "Ratio"
    );
    if llama_library.is_some() {
        println!("{:>8} {:>15} {:>15} {:>10}", "", "Ours", "llama.cpp", "");
    }

    for format in &formats {
        let our_gflops = benchmark_our_decode(
            &device,
            &queue,
            &our_library,
            format,
            NUM_HEADS,
            HEAD_DIM,
            512,
        );

        if let Some(ref llama_lib) = llama_library {
            let llama_gflops =
                benchmark_llama_decode(&device, &queue, llama_lib, format);
            let ratio = our_gflops / llama_gflops * 100.0;
            println!(
                "{:>8} {:>15.2} {:>15.2} {:>9.1}%",
                format, our_gflops, llama_gflops, ratio
            );
        } else {
            let time_us = (2.0
                * NUM_HEADS as f64
                * 512.0
                * HEAD_DIM as f64
                * NUM_ITER as f64)
                / our_gflops
                / 1e3;
            println!(
                "{:>8} {:>15.2} {:>15.2} {:>10}",
                format, our_gflops, time_us, "-"
            );
        }
    }

    // Benchmark different seq_k sizes
    println!("\n=== Seq_k Scaling (F16 decode) ===");
    let seq_ks = [128, 256, 512, 1024, 2048];

    println!(
        "{:>8} {:>12} {:>12} {:>8}",
        "seq_k", "Ours (GFLOPS)", "Time (µs)", "GB/s"
    );

    for seq_k in &seq_ks {
        let gflops = benchmark_our_decode(
            &device,
            &queue,
            &our_library,
            "f16",
            NUM_HEADS,
            HEAD_DIM,
            *seq_k,
        );
        let bytes = seq_k * NUM_HEADS * HEAD_DIM * 2; // F16 K + V
        let time_us = (2.0
            * NUM_HEADS as f64
            * *seq_k as f64
            * HEAD_DIM as f64
            * NUM_ITER as f64)
            / gflops
            / 1e3;
        let bandwidth = (bytes as f64 * NUM_ITER as f64) / time_us / 1e3;
        println!(
            "{:>8} {:>12.2} {:>12.2} {:>8.2}",
            seq_k, gflops, time_us, bandwidth
        );
    }

    // Prefill benchmark
    println!("\n=== Prefill (seq_q=16, seq_k=512) ===");
    println!("{:>8} {:>15} {:>15}", "Format", "Ours (GFLOPS)", "Time (µs)");

    for format in &formats {
        let gflops = benchmark_our_prefill(
            &device,
            &queue,
            &our_library,
            format,
            NUM_HEADS,
            HEAD_DIM,
            16,
            512,
        );
        let time_us = (2.0
            * NUM_HEADS as f64
            * 16.0
            * 512.0
            * HEAD_DIM as f64
            * NUM_ITER as f64)
            / gflops
            / 1e3;
        println!("{:>8} {:>15.2} {:>15.2}", format, gflops, time_us);
    }
}

fn load_llama_cpp_kernels() -> Result<String, std::io::Error> {
    // Load llama.cpp flash attention kernels
    let base = "llama.cpp-0.6.0/ggml/src/ggml-metal/kernels";

    let dequantize =
        std::fs::read_to_string(&format!("{}/dequantize.h", base))?;
    let fa_common =
        std::fs::read_to_string(&format!("{}/fa_common.metal", base))?;
    let fa_f16 = std::fs::read_to_string(&format!("{}/fa_f16.metal", base))?;
    let fa_q4_0 = std::fs::read_to_string(&format!("{}/fa_q4_0.metal", base))?;
    let fa_q4_1 = std::fs::read_to_string(&format!("{}/fa_q4_1.metal", base))?;
    let fa_q5_0 = std::fs::read_to_string(&format!("{}/fa_q5_0.metal", base))?;
    let fa_q5_1 = std::fs::read_to_string(&format!("{}/fa_q5_1.metal", base))?;
    let fa_q8_0 = std::fs::read_to_string(&format!("{}/fa_q8_0.metal", base))?;
    let fa_vec_f16 =
        std::fs::read_to_string(&format!("{}/fa_vec_f16.metal", base))?;
    let fa_vec_q4_0 =
        std::fs::read_to_string(&format!("{}/fa_vec_q4_0.metal", base))?;
    let fa_vec_q4_1 =
        std::fs::read_to_string(&format!("{}/fa_vec_q4_1.metal", base))?;
    let fa_vec_q5_0 =
        std::fs::read_to_string(&format!("{}/fa_vec_q5_0.metal", base))?;
    let fa_vec_q5_1 =
        std::fs::read_to_string(&format!("{}/fa_vec_q5_1.metal", base))?;
    let fa_vec_q8_0 =
        std::fs::read_to_string(&format!("{}/fa_vec_q8_0.metal", base))?;
    let fa_aux = std::fs::read_to_string(&format!("{}/fa_aux.metal", base))?;
    let fa_vec_common =
        std::fs::read_to_string(&format!("{}/fa_vec_common.metal", base))?;

    Ok(format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
        dequantize,
        fa_common,
        fa_f16,
        fa_q4_0,
        fa_q4_1,
        fa_q5_0,
        fa_q5_1,
        fa_q8_0,
        fa_vec_f16,
        fa_vec_q4_0,
        fa_vec_q4_1,
        fa_vec_q5_0,
        fa_vec_q5_1,
        fa_vec_q8_0,
        fa_aux,
        fa_vec_common
    ))
}

fn benchmark_our_decode(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
    num_heads: usize,
    head_dim: usize,
    seq_k: usize,
) -> f64 {
    let kernel_name = format!("flash_decode_attention_{}", format);
    let kernel = library.get_function(&kernel_name, None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&kernel).unwrap();

    // Create test data
    let q_size = num_heads * head_dim;
    let q_data: Vec<f32> = vec![0.5; q_size];

    let (k_quant, v_quant, k_row_size) =
        create_quantized_kv(format, num_heads, head_dim, seq_k);

    // Create buffers
    let q_buf = device
        .new_buffer((q_size * 4) as u64, MTLResourceOptions::StorageModeShared);
    let k_buf = device.new_buffer(
        k_quant.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer(
        v_quant.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device
        .new_buffer((q_size * 4) as u64, MTLResourceOptions::StorageModeShared);

    unsafe {
        std::ptr::copy_nonoverlapping(
            q_data.as_ptr(),
            q_buf.contents() as *mut f32,
            q_size,
        );
        std::ptr::copy_nonoverlapping(
            k_quant.as_ptr(),
            k_buf.contents() as *mut u8,
            k_quant.len(),
        );
        std::ptr::copy_nonoverlapping(
            v_quant.as_ptr(),
            v_buf.contents() as *mut u8,
            v_quant.len(),
        );
    }

    // Arguments
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
        heads: num_heads as u32,
        seq_q: 1,
        seq_k: seq_k as u32,
        head_dim: head_dim as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
        q_stride_b: (num_heads * head_dim) as u32,
        q_stride_h: head_dim as u32,
        q_stride_s: head_dim as u32,
        k_stride_b: (num_heads * seq_k * k_row_size) as u32,
        k_stride_h: (seq_k * k_row_size) as u32,
        k_stride_s: k_row_size as u32,
        v_stride_b: (num_heads * seq_k * k_row_size) as u32,
        v_stride_h: (seq_k * k_row_size) as u32,
        v_stride_s: k_row_size as u32,
        o_stride_b: (num_heads * head_dim) as u32,
        o_stride_h: head_dim as u32,
        o_stride_s: head_dim as u32,
    };

    // Warmup
    for _ in 0..WARMUP {
        let args_bytes = unsafe {
            std::slice::from_raw_parts(
                &args as *const _ as *const u8,
                std::mem::size_of::<Args>(),
            )
        };
        run_decode_kernel(
            &queue, &pipeline, args_bytes, &q_buf, &k_buf, &v_buf, &out_buf,
            num_heads,
        );
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..NUM_ITER {
        let args_bytes = unsafe {
            std::slice::from_raw_parts(
                &args as *const _ as *const u8,
                std::mem::size_of::<Args>(),
            )
        };
        run_decode_kernel(
            &queue, &pipeline, args_bytes, &q_buf, &k_buf, &v_buf, &out_buf,
            num_heads,
        );
    }
    let elapsed = start.elapsed();

    let flops = 2.0
        * num_heads as f64
        * seq_k as f64
        * head_dim as f64
        * NUM_ITER as f64;
    flops / elapsed.as_secs_f64() / 1e9
}

fn run_decode_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    args: &[u8],
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    out_buf: &Buffer,
    num_heads: usize,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(0, args.len() as u64, args.as_ptr() as *const _);
    enc.set_buffer(1, Some(q_buf), 0);
    enc.set_buffer(2, Some(k_buf), 0);
    enc.set_buffer(3, Some(v_buf), 0);
    enc.set_buffer(4, Some(out_buf), 0);

    let grid_size = MTLSize { width: 1, height: num_heads as u64, depth: 1 };
    let tg_size = MTLSize { width: 32, height: 1, depth: 1 };
    enc.dispatch_thread_groups(grid_size, tg_size);

    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn benchmark_llama_decode(
    _device: &Device,
    _queue: &CommandQueue,
    _library: &Library,
    _format: &str,
) -> f64 {
    // llama.cpp uses different kernel interface with more complex stride handling
    // For now, return 0 to indicate we need to implement this
    // The kernels require specific tensor layout with ne*, nb* parameters
    0.0
}

fn benchmark_our_prefill(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
    num_heads: usize,
    head_dim: usize,
    seq_q: usize,
    seq_k: usize,
) -> f64 {
    let kernel_name = format!("flash_prefill_attention_{}", format);
    let kernel = library.get_function(&kernel_name, None).unwrap();
    let pipeline =
        device.new_compute_pipeline_state_with_function(&kernel).unwrap();

    let q_size = seq_q * num_heads * head_dim;
    let q_data: Vec<f32> = vec![0.5; q_size];

    let (k_quant, v_quant, k_row_size) =
        create_quantized_kv(format, num_heads, head_dim, seq_k);

    let q_buf = device
        .new_buffer((q_size * 4) as u64, MTLResourceOptions::StorageModeShared);
    let k_buf = device.new_buffer(
        k_quant.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let v_buf = device.new_buffer(
        v_quant.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device
        .new_buffer((q_size * 4) as u64, MTLResourceOptions::StorageModeShared);

    unsafe {
        std::ptr::copy_nonoverlapping(
            q_data.as_ptr(),
            q_buf.contents() as *mut f32,
            q_size,
        );
        std::ptr::copy_nonoverlapping(
            k_quant.as_ptr(),
            k_buf.contents() as *mut u8,
            k_quant.len(),
        );
        std::ptr::copy_nonoverlapping(
            v_quant.as_ptr(),
            v_buf.contents() as *mut u8,
            v_quant.len(),
        );
    }

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
        heads: num_heads as u32,
        seq_q: seq_q as u32,
        seq_k: seq_k as u32,
        head_dim: head_dim as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
        q_stride_b: (num_heads * seq_q * head_dim) as u32,
        q_stride_h: (seq_q * head_dim) as u32,
        q_stride_s: head_dim as u32,
        k_stride_b: (num_heads * seq_k * k_row_size) as u32,
        k_stride_h: (seq_k * k_row_size) as u32,
        k_stride_s: k_row_size as u32,
        v_stride_b: (num_heads * seq_k * k_row_size) as u32,
        v_stride_h: (seq_k * k_row_size) as u32,
        v_stride_s: k_row_size as u32,
        o_stride_b: (num_heads * seq_q * head_dim) as u32,
        o_stride_h: (seq_q * head_dim) as u32,
        o_stride_s: head_dim as u32,
    };

    // Warmup
    for _ in 0..WARMUP {
        let args_bytes = unsafe {
            std::slice::from_raw_parts(
                &args as *const _ as *const u8,
                std::mem::size_of::<Args>(),
            )
        };
        run_prefill_kernel(
            &queue, &pipeline, args_bytes, &q_buf, &k_buf, &v_buf, &out_buf,
            num_heads, seq_q,
        );
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..NUM_ITER {
        let args_bytes = unsafe {
            std::slice::from_raw_parts(
                &args as *const _ as *const u8,
                std::mem::size_of::<Args>(),
            )
        };
        run_prefill_kernel(
            &queue, &pipeline, args_bytes, &q_buf, &k_buf, &v_buf, &out_buf,
            num_heads, seq_q,
        );
    }
    let elapsed = start.elapsed();

    let flops = 2.0
        * num_heads as f64
        * seq_q as f64
        * seq_k as f64
        * head_dim as f64
        * NUM_ITER as f64;
    flops / elapsed.as_secs_f64() / 1e9
}

fn run_prefill_kernel(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    args: &[u8],
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    out_buf: &Buffer,
    num_heads: usize,
    seq_q: usize,
) {
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_bytes(0, args.len() as u64, args.as_ptr() as *const _);
    enc.set_buffer(1, Some(q_buf), 0);
    enc.set_buffer(2, Some(k_buf), 0);
    enc.set_buffer(3, Some(v_buf), 0);
    enc.set_buffer(4, Some(out_buf), 0);

    let grid_size =
        MTLSize { width: seq_q as u64, height: num_heads as u64, depth: 1 };
    let tg_size = MTLSize { width: 128, height: 1, depth: 1 };
    enc.dispatch_thread_groups(grid_size, tg_size);

    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
}

fn create_quantized_kv(
    format: &str,
    num_heads: usize,
    head_dim: usize,
    seq_k: usize,
) -> (Vec<u8>, Vec<u8>, usize) {
    let k_size = seq_k * num_heads * head_dim;
    let k_data: Vec<f32> = vec![0.3; k_size];
    let v_data: Vec<f32> = vec![0.2; k_size];

    match format {
        "f16" => {
            let mut k_quant = Vec::with_capacity(k_size * 2);
            let mut v_quant = Vec::with_capacity(k_size * 2);
            for &v in &k_data {
                let bits = half::f16::from_f32(v).to_bits();
                k_quant.extend_from_slice(&bits.to_ne_bytes());
            }
            for &v in &v_data {
                let bits = half::f16::from_f32(v).to_bits();
                v_quant.extend_from_slice(&bits.to_ne_bytes());
            }
            (k_quant, v_quant, head_dim * 2)
        }
        "q4_0" => {
            let blocks_per_row = head_dim / 32;
            let row_size = blocks_per_row * std::mem::size_of::<BlockQ4_0>();
            let (k, v) =
                quantize_kv_q4_0(&k_data, &v_data, num_heads, head_dim, seq_k);
            (k, v, row_size)
        }
        "q4_1" => {
            let blocks_per_row = head_dim / 32;
            let row_size = blocks_per_row * std::mem::size_of::<BlockQ4_1>();
            let (k, v) =
                quantize_kv_q4_1(&k_data, &v_data, num_heads, head_dim, seq_k);
            (k, v, row_size)
        }
        "q5_0" => {
            let blocks_per_row = head_dim / 32;
            let row_size = blocks_per_row * std::mem::size_of::<BlockQ5_0>();
            let (k, v) =
                quantize_kv_q5_0(&k_data, &v_data, num_heads, head_dim, seq_k);
            (k, v, row_size)
        }
        "q5_1" => {
            let blocks_per_row = head_dim / 32;
            let row_size = blocks_per_row * std::mem::size_of::<BlockQ5_1>();
            let (k, v) =
                quantize_kv_q5_1(&k_data, &v_data, num_heads, head_dim, seq_k);
            (k, v, row_size)
        }
        "q8_0" => {
            let blocks_per_row = head_dim / 32;
            let row_size = blocks_per_row * std::mem::size_of::<BlockQ8_0>();
            let (k, v) =
                quantize_kv_q8_0(&k_data, &v_data, num_heads, head_dim, seq_k);
            (k, v, row_size)
        }
        _ => panic!("Unknown format: {}", format),
    }
}

fn quantize_kv_q4_0(
    k_data: &[f32],
    v_data: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_k: usize,
) -> (Vec<u8>, Vec<u8>) {
    let blocks_per_row = head_dim / 32;
    let row_size = blocks_per_row * std::mem::size_of::<BlockQ4_0>();
    let mut k_quant = vec![0u8; num_heads * seq_k * row_size];
    let mut v_quant = vec![0u8; num_heads * seq_k * row_size];

    for h in 0..num_heads {
        for s in 0..seq_k {
            let offset = h * seq_k * head_dim + s * head_dim;
            let k_row = &k_data[offset..offset + head_dim];
            let v_row = &v_data[offset..offset + head_dim];

            let k_out = &mut k_quant[(h * seq_k + s) * row_size..];
            let v_out = &mut v_quant[(h * seq_k + s) * row_size..];

            let mut k_blocks: Vec<BlockQ4_0> =
                vec![BlockQ4_0 { d: 0, qs: [0; 16] }; blocks_per_row];
            let mut v_blocks: Vec<BlockQ4_0> =
                vec![BlockQ4_0 { d: 0, qs: [0; 16] }; blocks_per_row];
            unsafe {
                quantize_row_q4_0(k_row, &mut k_blocks);
                quantize_row_q4_0(v_row, &mut v_blocks);
            }

            k_out[..row_size].copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    k_blocks.as_ptr() as *const u8,
                    row_size,
                )
            });
            v_out[..row_size].copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    v_blocks.as_ptr() as *const u8,
                    row_size,
                )
            });
        }
    }
    (k_quant, v_quant)
}

fn quantize_kv_q4_1(
    k_data: &[f32],
    v_data: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_k: usize,
) -> (Vec<u8>, Vec<u8>) {
    let blocks_per_row = head_dim / 32;
    let row_size = blocks_per_row * std::mem::size_of::<BlockQ4_1>();
    let mut k_quant = vec![0u8; num_heads * seq_k * row_size];
    let mut v_quant = vec![0u8; num_heads * seq_k * row_size];

    for h in 0..num_heads {
        for s in 0..seq_k {
            let offset = h * seq_k * head_dim + s * head_dim;
            let k_row = &k_data[offset..offset + head_dim];
            let v_row = &v_data[offset..offset + head_dim];

            let k_out = &mut k_quant[(h * seq_k + s) * row_size..];
            let v_out = &mut v_quant[(h * seq_k + s) * row_size..];

            let mut k_blocks: Vec<BlockQ4_1> =
                vec![BlockQ4_1 { d: 0, m: 0, qs: [0; 16] }; blocks_per_row];
            let mut v_blocks: Vec<BlockQ4_1> =
                vec![BlockQ4_1 { d: 0, m: 0, qs: [0; 16] }; blocks_per_row];
            unsafe {
                quantize_row_q4_1(k_row, &mut k_blocks);
                quantize_row_q4_1(v_row, &mut v_blocks);
            }

            k_out[..row_size].copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    k_blocks.as_ptr() as *const u8,
                    row_size,
                )
            });
            v_out[..row_size].copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    v_blocks.as_ptr() as *const u8,
                    row_size,
                )
            });
        }
    }
    (k_quant, v_quant)
}

fn quantize_kv_q5_0(
    k_data: &[f32],
    v_data: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_k: usize,
) -> (Vec<u8>, Vec<u8>) {
    let blocks_per_row = head_dim / 32;
    let row_size = blocks_per_row * std::mem::size_of::<BlockQ5_0>();
    let mut k_quant = vec![0u8; num_heads * seq_k * row_size];
    let mut v_quant = vec![0u8; num_heads * seq_k * row_size];

    for h in 0..num_heads {
        for s in 0..seq_k {
            let offset = h * seq_k * head_dim + s * head_dim;
            let k_row = &k_data[offset..offset + head_dim];
            let v_row = &v_data[offset..offset + head_dim];

            let k_out = &mut k_quant[(h * seq_k + s) * row_size..];
            let v_out = &mut v_quant[(h * seq_k + s) * row_size..];

            let mut k_blocks: Vec<BlockQ5_0> =
                vec![
                    BlockQ5_0 { d: 0, qh: [0; 4], qs: [0; 16] };
                    blocks_per_row
                ];
            let mut v_blocks: Vec<BlockQ5_0> =
                vec![
                    BlockQ5_0 { d: 0, qh: [0; 4], qs: [0; 16] };
                    blocks_per_row
                ];
            unsafe {
                quantize_row_q5_0(k_row, &mut k_blocks);
                quantize_row_q5_0(v_row, &mut v_blocks);
            }

            k_out[..row_size].copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    k_blocks.as_ptr() as *const u8,
                    row_size,
                )
            });
            v_out[..row_size].copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    v_blocks.as_ptr() as *const u8,
                    row_size,
                )
            });
        }
    }
    (k_quant, v_quant)
}

fn quantize_kv_q5_1(
    k_data: &[f32],
    v_data: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_k: usize,
) -> (Vec<u8>, Vec<u8>) {
    let blocks_per_row = head_dim / 32;
    let row_size = blocks_per_row * std::mem::size_of::<BlockQ5_1>();
    let mut k_quant = vec![0u8; num_heads * seq_k * row_size];
    let mut v_quant = vec![0u8; num_heads * seq_k * row_size];

    for h in 0..num_heads {
        for s in 0..seq_k {
            let offset = h * seq_k * head_dim + s * head_dim;
            let k_row = &k_data[offset..offset + head_dim];
            let v_row = &v_data[offset..offset + head_dim];

            let k_out = &mut k_quant[(h * seq_k + s) * row_size..];
            let v_out = &mut v_quant[(h * seq_k + s) * row_size..];

            let mut k_blocks: Vec<BlockQ5_1> =
                vec![
                    BlockQ5_1 { d: 0, m: 0, qh: [0; 4], qs: [0; 16] };
                    blocks_per_row
                ];
            let mut v_blocks: Vec<BlockQ5_1> =
                vec![
                    BlockQ5_1 { d: 0, m: 0, qh: [0; 4], qs: [0; 16] };
                    blocks_per_row
                ];
            unsafe {
                quantize_row_q5_1(k_row, &mut k_blocks);
                quantize_row_q5_1(v_row, &mut v_blocks);
            }

            k_out[..row_size].copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    k_blocks.as_ptr() as *const u8,
                    row_size,
                )
            });
            v_out[..row_size].copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    v_blocks.as_ptr() as *const u8,
                    row_size,
                )
            });
        }
    }
    (k_quant, v_quant)
}

fn quantize_kv_q8_0(
    k_data: &[f32],
    v_data: &[f32],
    num_heads: usize,
    head_dim: usize,
    seq_k: usize,
) -> (Vec<u8>, Vec<u8>) {
    let blocks_per_row = head_dim / 32;
    let row_size = blocks_per_row * std::mem::size_of::<BlockQ8_0>();
    let mut k_quant = vec![0u8; num_heads * seq_k * row_size];
    let mut v_quant = vec![0u8; num_heads * seq_k * row_size];

    for h in 0..num_heads {
        for s in 0..seq_k {
            let offset = h * seq_k * head_dim + s * head_dim;
            let k_row = &k_data[offset..offset + head_dim];
            let v_row = &v_data[offset..offset + head_dim];

            let k_out = &mut k_quant[(h * seq_k + s) * row_size..];
            let v_out = &mut v_quant[(h * seq_k + s) * row_size..];

            let mut k_blocks: Vec<BlockQ8_0> =
                vec![BlockQ8_0 { d: 0, qs: [0; 32] }; blocks_per_row];
            let mut v_blocks: Vec<BlockQ8_0> =
                vec![BlockQ8_0 { d: 0, qs: [0; 32] }; blocks_per_row];
            unsafe {
                quantize_row_q8_0(k_row, &mut k_blocks);
                quantize_row_q8_0(v_row, &mut v_blocks);
            }

            k_out[..row_size].copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    k_blocks.as_ptr() as *const u8,
                    row_size,
                )
            });
            v_out[..row_size].copy_from_slice(unsafe {
                std::slice::from_raw_parts(
                    v_blocks.as_ptr() as *const u8,
                    row_size,
                )
            });
        }
    }
    (k_quant, v_quant)
}
