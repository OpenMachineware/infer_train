// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Comprehensive performance benchmark for all flash decode formats
// Measures GFLOPS across all formats and seq_k sizes

use half::f16;
use infer_train::quant::kv_cache::{
    quantize_row_q4_0, quantize_row_q4_1, quantize_row_q5_0, quantize_row_q5_1,
    quantize_row_q8_0, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0,
};
use metal::*;
use std::time::Instant;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const WARMUP: usize = 50;
const ITERATIONS: usize = 200;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source =
        include_str!("../src/ops/metal/attention/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).unwrap();

    println!("=== Flash Decode Performance Benchmark ===");
    println!(
        "Config: {} heads, {} head_dim, {} warmup, {} iterations\n",
        NUM_HEADS, HEAD_DIM, WARMUP, ITERATIONS
    );

    let formats = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "bf16"];
    let seq_ks = [256, 512, 1024, 2048, 4096];
    let groups = [1, 2, 4, 8];

    println!("Format   seq_k   g1      g2      g4      g8      Best");
    println!("{}", "-".repeat(60));

    for format in &formats {
        for seq_k in &seq_ks {
            let mut best_gflops = 0.0;
            let mut best_group = 0;
            let mut results = Vec::new();

            for nsg in &groups {
                let gflops = benchmark_format(
                    &device, &queue, &library, format, *seq_k, *nsg,
                );
                results.push(format!("{:.2}", gflops));
                if gflops > best_gflops {
                    best_gflops = gflops;
                    best_group = *nsg;
                }
            }

            println!(
                "{:8} {:5}  {:>7} {:>7} {:>7} {:>7}  g{} ({:.2} GFLOPS)",
                format,
                seq_k,
                results[0],
                results[1],
                results[2],
                results[3],
                best_group,
                best_gflops
            );
        }
        println!();
    }
}

fn benchmark_format(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
    seq_k: usize,
    nsg: usize,
) -> f64 {
    // Create test data
    let q_data: Vec<f32> =
        (0..NUM_HEADS * HEAD_DIM).map(|i| (i % 10) as f32 * 0.1).collect();
    let k_raw: Vec<f32> =
        (0..seq_k * HEAD_DIM).map(|i| (i % 20) as f32 * 0.05).collect();
    let v_raw: Vec<f32> =
        (0..seq_k * HEAD_DIM).map(|i| (i % 15) as f32 * 0.03).collect();

    // Quantize K/V
    let (k_quant, v_quant, k_stride, v_stride) =
        quantize_kv(format, &k_raw, &v_raw, seq_k);

    // Get kernel
    let kernel_name = format!("flash_decode_multi_group_{}_g{}", format, nsg);
    let kernel = match library.get_function(&kernel_name, None) {
        Ok(k) => k,
        Err(_) => return 0.0,
    };
    let pso = device.new_compute_pipeline_state_with_function(&kernel).unwrap();

    // Create buffers
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

    // Args
    #[repr(C)]
    struct FlashAttnArgs {
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

    let args = FlashAttnArgs {
        batch: 1,
        heads: NUM_HEADS as u32,
        seq_q: 1,
        seq_k: seq_k as u32,
        head_dim: HEAD_DIM as u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        q_stride_b: (NUM_HEADS * HEAD_DIM) as u32,
        q_stride_h: HEAD_DIM as u32,
        q_stride_s: HEAD_DIM as u32,
        k_stride_b: (seq_k * k_stride) as u32,
        k_stride_h: (seq_k * k_stride) as u32,
        k_stride_s: k_stride as u32,
        v_stride_b: (seq_k * v_stride) as u32,
        v_stride_h: (seq_k * v_stride) as u32,
        v_stride_s: v_stride as u32,
        o_stride_b: (NUM_HEADS * HEAD_DIM) as u32,
        o_stride_h: HEAD_DIM as u32,
        o_stride_s: HEAD_DIM as u32,
    };

    let args_bytes = unsafe {
        std::slice::from_raw_parts(
            &args as *const _ as *const u8,
            std::mem::size_of::<FlashAttnArgs>(),
        )
    };

    // Threadgroup memory
    let query_size = 32 * 16;
    let ms_size = nsg * 16;
    let out_size = nsg * 4 * 8 * 16;
    let shmem_size = query_size + ms_size + out_size;

    // Warmup
    for _ in 0..WARMUP {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_bytes(
            0,
            args_bytes.len() as u64,
            args_bytes.as_ptr() as *const _,
        );
        enc.set_buffer(1, Some(&q_buf), 0);
        enc.set_buffer(2, Some(&k_buf), 0);
        enc.set_buffer(3, Some(&v_buf), 0);
        enc.set_buffer(4, Some(&out_buf), 0);
        enc.set_threadgroup_memory_length(0, shmem_size as u64);
        enc.dispatch_thread_groups(
            MTLSize::new(1, NUM_HEADS as u64, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_bytes(
            0,
            args_bytes.len() as u64,
            args_bytes.as_ptr() as *const _,
        );
        enc.set_buffer(1, Some(&q_buf), 0);
        enc.set_buffer(2, Some(&k_buf), 0);
        enc.set_buffer(3, Some(&v_buf), 0);
        enc.set_buffer(4, Some(&out_buf), 0);
        enc.set_threadgroup_memory_length(0, shmem_size as u64);
        enc.dispatch_thread_groups(
            MTLSize::new(1, NUM_HEADS as u64, 1),
            MTLSize::new(32, nsg as u64, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let elapsed = start.elapsed();

    // Calculate GFLOPS
    // FLOPs per token: 2 * head_dim (Q·K) + 2 * head_dim (weighted V) = 4 * head_dim
    // Total FLOPs: num_heads * seq_k * 4 * head_dim
    let flops_per_iter = NUM_HEADS * seq_k * 4 * HEAD_DIM;
    let total_flops = flops_per_iter * ITERATIONS;
    let gflops = total_flops as f64 / elapsed.as_secs_f64() / 1e9;

    gflops
}

fn quantize_kv(
    format: &str,
    k_f32: &[f32],
    v_f32: &[f32],
    seq_k: usize,
) -> (Vec<u8>, Vec<u8>, usize, usize) {
    let head_dim = HEAD_DIM;

    match format {
        "q4_0" => {
            let k = quantize_to_q4_0(k_f32, seq_k, head_dim);
            let v = quantize_to_q4_0(v_f32, seq_k, head_dim);
            let stride = (head_dim / 32) * std::mem::size_of::<BlockQ4_0>();
            (k, v, stride, stride)
        }
        "q4_1" => {
            let k = quantize_to_q4_1(k_f32, seq_k, head_dim);
            let v = quantize_to_q4_1(v_f32, seq_k, head_dim);
            let stride = (head_dim / 32) * std::mem::size_of::<BlockQ4_1>();
            (k, v, stride, stride)
        }
        "q5_0" => {
            let k = quantize_to_q5_0(k_f32, seq_k, head_dim);
            let v = quantize_to_q5_0(v_f32, seq_k, head_dim);
            let stride = (head_dim / 32) * std::mem::size_of::<BlockQ5_0>();
            (k, v, stride, stride)
        }
        "q5_1" => {
            let k = quantize_to_q5_1(k_f32, seq_k, head_dim);
            let v = quantize_to_q5_1(v_f32, seq_k, head_dim);
            let stride = (head_dim / 32) * std::mem::size_of::<BlockQ5_1>();
            (k, v, stride, stride)
        }
        "q8_0" => {
            let k = quantize_to_q8_0(k_f32, seq_k, head_dim);
            let v = quantize_to_q8_0(v_f32, seq_k, head_dim);
            let stride = (head_dim / 32) * std::mem::size_of::<BlockQ8_0>();
            (k, v, stride, stride)
        }
        "bf16" => {
            let k = quantize_to_bf16(k_f32, seq_k, head_dim);
            let v = quantize_to_bf16(v_f32, seq_k, head_dim);
            let stride = head_dim * 2;
            (k, v, stride, stride)
        }
        _ => panic!("Unknown format: {}", format),
    }
}

fn quantize_to_q4_0(data: &[f32], seq: usize, head_dim: usize) -> Vec<u8> {
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

fn quantize_to_q4_1(data: &[f32], seq: usize, head_dim: usize) -> Vec<u8> {
    let blocks_per_row = head_dim / 32;
    let mut result = Vec::with_capacity(
        seq * blocks_per_row * std::mem::size_of::<BlockQ4_1>(),
    );

    for s in 0..seq {
        let row_data = &data[s * head_dim..(s + 1) * head_dim];
        let mut blocks: Vec<BlockQ4_1> =
            vec![unsafe { std::mem::zeroed() }; blocks_per_row];
        quantize_row_q4_1(row_data, &mut blocks);
        let block_bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                blocks.len() * std::mem::size_of::<BlockQ4_1>(),
            )
        };
        result.extend_from_slice(block_bytes);
    }
    result
}

fn quantize_to_q5_0(data: &[f32], seq: usize, head_dim: usize) -> Vec<u8> {
    let blocks_per_row = head_dim / 32;
    let mut result = Vec::with_capacity(
        seq * blocks_per_row * std::mem::size_of::<BlockQ5_0>(),
    );

    for s in 0..seq {
        let row_data = &data[s * head_dim..(s + 1) * head_dim];
        let mut blocks: Vec<BlockQ5_0> =
            vec![unsafe { std::mem::zeroed() }; blocks_per_row];
        quantize_row_q5_0(row_data, &mut blocks);
        let block_bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                blocks.len() * std::mem::size_of::<BlockQ5_0>(),
            )
        };
        result.extend_from_slice(block_bytes);
    }
    result
}

fn quantize_to_q5_1(data: &[f32], seq: usize, head_dim: usize) -> Vec<u8> {
    let blocks_per_row = head_dim / 32;
    let mut result = Vec::with_capacity(
        seq * blocks_per_row * std::mem::size_of::<BlockQ5_1>(),
    );

    for s in 0..seq {
        let row_data = &data[s * head_dim..(s + 1) * head_dim];
        let mut blocks: Vec<BlockQ5_1> =
            vec![unsafe { std::mem::zeroed() }; blocks_per_row];
        quantize_row_q5_1(row_data, &mut blocks);
        let block_bytes = unsafe {
            std::slice::from_raw_parts(
                blocks.as_ptr() as *const u8,
                blocks.len() * std::mem::size_of::<BlockQ5_1>(),
            )
        };
        result.extend_from_slice(block_bytes);
    }
    result
}

fn quantize_to_q8_0(data: &[f32], seq: usize, head_dim: usize) -> Vec<u8> {
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

fn quantize_to_bf16(data: &[f32], seq: usize, head_dim: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(seq * head_dim * 2);
    for &v in data.iter().take(seq * head_dim) {
        let bf16 = bf16_from_f32(v);
        result.extend_from_slice(&bf16.to_le_bytes());
    }
    result
}

// Simple BF16 conversion
fn bf16_from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 31) & 1;
    let exponent = (bits >> 23) & 0xFF;
    let mantissa = bits & 0x7FFFFF;

    if exponent == 0 {
        // Zero or subnormal - round to zero
        if sign == 0 {
            0
        } else {
            0x8000
        }
    } else if exponent == 0xFF {
        // Infinity or NaN
        if mantissa == 0 {
            if sign == 0 {
                0x7F80
            } else {
                0xFF80
            }
        } else {
            // Preserve NaN
            0x7FC0 | (mantissa >> 21) as u16
        }
    } else {
        // Normal number - round to nearest even
        let new_exp = exponent as i32 - 127 + 127; // No bias change for BF16
        let new_mantissa = mantissa >> 16;
        let rounding_bit = (mantissa >> 15) & 1;

        let mut result =
            ((sign << 15) | ((new_exp as u32) << 7) | new_mantissa) as u16;

        // Round to nearest even
        if rounding_bit == 1 {
            if (mantissa & 0x7FFF) != 0 || (new_mantissa & 1) == 1 {
                result += 1;
            }
        }

        result
    }
}
