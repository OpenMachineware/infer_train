// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test multi-SIMD group flash attention for all quantized formats

use half::f16;
use infer_train::quant::kv_cache::{
    BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0,
};
use metal::*;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    // Load our kernels
    let our_source =
        include_str!("../src/ops/metal/attention/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let our_library =
        device.new_library_with_source(our_source, &options).unwrap();

    println!("=== Multi-SIMD Group All Formats Test ===\n");

    let formats = ["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "bf16"];

    for format in &formats {
        test_format(&device, &queue, &our_library, format);
    }
}

fn test_format(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    format: &str,
) {
    println!("=== Format: {} ===", format);

    let num_heads = 32;
    let head_dim = 128;
    let seq_k = 256;

    // Create test data
    let q_data: Vec<f32> =
        (0..num_heads * head_dim).map(|i| (i % 10) as f32 * 0.1).collect();
    let k_raw: Vec<f32> =
        (0..seq_k * head_dim).map(|i| (i % 20) as f32 * 0.05).collect();
    let v_raw: Vec<f32> =
        (0..seq_k * head_dim).map(|i| (i % 15) as f32 * 0.03).collect();

    // Quantize K/V
    let (k_quant, v_quant, k_stride, v_stride) =
        quantize_kv(format, &k_raw, &v_raw, seq_k, head_dim);

    // CPU reference (using F16 K/V for consistency with quantization)
    let k_f16: Vec<f16> = k_raw.iter().map(|&x| f16::from_f32(x)).collect();
    let v_f16: Vec<f16> = v_raw.iter().map(|&x| f16::from_f32(x)).collect();
    let ref_output = compute_cpu_attention(
        &q_data, &k_f16, &v_f16, num_heads, head_dim, seq_k,
    );

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
        (num_heads * head_dim * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Test all group counts
    for nsg in [1, 2, 4, 8] {
        let kernel_name =
            format!("flash_decode_multi_group_{}_g{}", format, nsg);
        let kernel = match library.get_function(&kernel_name, None) {
            Ok(k) => k,
            Err(_) => {
                println!("  g{}: SKIP (kernel not found)", nsg);
                continue;
            }
        };
        let pso =
            device.new_compute_pipeline_state_with_function(&kernel).unwrap();

        // Clear output
        unsafe {
            std::ptr::write_bytes(
                out_buf.contents() as *mut f32,
                0,
                num_heads * head_dim,
            );
        }

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
            heads: num_heads as u32,
            seq_q: 1,
            seq_k: seq_k as u32,
            head_dim: head_dim as u32,
            scale: 1.0 / (head_dim as f32).sqrt(),
            q_stride_b: (num_heads * head_dim) as u32,
            q_stride_h: head_dim as u32,
            q_stride_s: head_dim as u32,
            k_stride_b: (seq_k * k_stride) as u32, // Total K for one batch
            k_stride_h: (seq_k * k_stride) as u32, // All tokens for one head
            k_stride_s: k_stride as u32,           // One token
            v_stride_b: (seq_k * v_stride) as u32,
            v_stride_h: (seq_k * v_stride) as u32,
            v_stride_s: v_stride as u32,
            o_stride_b: (num_heads * head_dim) as u32,
            o_stride_h: head_dim as u32,
            o_stride_s: head_dim as u32,
        };

        let args_bytes = unsafe {
            std::slice::from_raw_parts(
                &args as *const _ as *const u8,
                std::mem::size_of::<FlashAttnArgs>(),
            )
        };

        // Threadgroup memory
        // Layout: [query (512B)] [M+S per group (16B each)] [output per group (512B each)]
        let query_size = 32 * 16;
        let ms_size = nsg * 16;
        let out_size = nsg * 4 * 8 * 16;
        let shmem_size = query_size + ms_size + out_size;

        // Dispatch
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

        // Check result (only compare first head)
        let gpu_output = unsafe {
            std::slice::from_raw_parts(
                out_buf.contents() as *const f32,
                num_heads * head_dim,
            )
        };

        let max_diff = (0..head_dim)
            .map(|i| (gpu_output[i] - ref_output[i]).abs())
            .fold(0.0f32, |a, b| a.max(b));

        println!(
            "  g{}: max_diff = {:.6e} {}",
            nsg,
            max_diff,
            if max_diff < 1e-2 { "✓" } else { "✗" }
        );
    }
    println!();
}

fn compute_cpu_attention(
    q: &[f32],
    k_f16: &[f16],
    v_f16: &[f16],
    num_heads: usize,
    head_dim: usize,
    seq_k: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; head_dim];

    // Compute for first head
    let q_head = &q[0..head_dim];

    // Online softmax with weighted V accumulation
    let mut max_val = f32::NEG_INFINITY;
    let mut sum_val = 0.0f32;

    for k in 0..seq_k {
        // Compute Q·K^T for this position
        let mut score = 0.0f32;
        for d in 0..head_dim {
            score += q_head[d] * k_f16[k * head_dim + d].to_f32();
        }
        score *= scale;

        // Online softmax update
        let old_max = max_val;
        max_val = max_val.max(score);

        if old_max > f32::NEG_INFINITY / 2.0 {
            let scale_factor = (old_max - max_val).exp();
            sum_val *= scale_factor;
            for d in 0..head_dim {
                output[d] *= scale_factor;
            }
        }

        let weight = (score - max_val).exp();
        sum_val += weight;

        for d in 0..head_dim {
            output[d] += weight * v_f16[k * head_dim + d].to_f32();
        }
    }

    // Normalize
    for d in 0..head_dim {
        output[d] /= sum_val;
    }

    output
}

fn quantize_kv(
    format: &str,
    k_f32: &[f32],
    v_f32: &[f32],
    seq_k: usize,
    head_dim: usize,
) -> (Vec<u8>, Vec<u8>, usize, usize) {
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
    use infer_train::quant::kv_cache::{BlockQ4_0, quantize_row_q4_0};
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
    use infer_train::quant::kv_cache::{BlockQ4_1, quantize_row_q4_1};
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
    use infer_train::quant::kv_cache::{BlockQ5_0, quantize_row_q5_0};
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
    use infer_train::quant::kv_cache::{BlockQ5_1, quantize_row_q5_1};
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
    use infer_train::quant::kv_cache::{BlockQ8_0, quantize_row_q8_0};
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
    // BF16: just reinterpret f32 as bf16 (truncate lower 16 bits)
    let mut result = Vec::with_capacity(seq * head_dim * 2);
    for &f in data.iter() {
        let bits = f.to_bits();
        let bf16_bits = (bits >> 16) as u16;
        result.extend_from_slice(&bf16_bits.to_ne_bytes());
    }
    result
}
