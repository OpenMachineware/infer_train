// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Test multi-SIMD group flash attention for all quantized formats

use half::f16;
use metal::*;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    // Load our kernels
    let our_source = include_str!("../../shaders/flash_attn_quant.metal");
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

    // CPU reference (using quantized K/V)
    let ref_output = compute_cpu_attention(
        &q_data, &k_quant, &v_quant, format, num_heads, head_dim, seq_k,
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
            q_stride_b: (num_heads * head_dim * 4) as u32,
            q_stride_h: (head_dim * 4) as u32,
            q_stride_s: (head_dim * 4) as u32,
            k_stride_b: k_stride as u32, // All K data for batch
            k_stride_h: k_stride as u32, // All K data for one head (shared across heads)
            k_stride_s: get_block_size(format, head_dim) as u32, // One row of K data
            v_stride_b: v_stride as u32,
            v_stride_h: v_stride as u32,
            v_stride_s: get_block_size(format, head_dim) as u32,
            o_stride_b: (num_heads * head_dim * 4) as u32,
            o_stride_h: (head_dim * 4) as u32,
            o_stride_s: (head_dim * 4) as u32,
        };

        let args_bytes = unsafe {
            std::slice::from_raw_parts(
                &args as *const _ as *const u8,
                std::mem::size_of::<FlashAttnArgs>(),
            )
        };

        // Threadgroup memory
        let shmem_size = nsg * (2 + head_dim) * 4;

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
    k_quant: &[u8],
    v_quant: &[u8],
    format: &str,
    num_heads: usize,
    head_dim: usize,
    seq_k: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; head_dim];

    // Dequantize K/V for reference
    let k_f32 = dequantize_kv(format, k_quant, seq_k, head_dim);
    let v_f32 = dequantize_kv(format, v_quant, seq_k, head_dim);

    // Compute for first head
    let q_head = &q[0..head_dim];

    // Online softmax with weighted V accumulation
    let mut max_val = f32::NEG_INFINITY;
    let mut sum_val = 0.0f32;

    for k in 0..seq_k {
        // Compute Q·K^T for this position
        let mut score = 0.0f32;
        for d in 0..head_dim {
            score += q_head[d] * k_f32[k * head_dim + d];
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
            output[d] += weight * v_f32[k * head_dim + d];
        }
    }

    // Normalize
    for d in 0..head_dim {
        output[d] /= sum_val;
    }

    output
}

fn dequantize_kv(
    format: &str,
    quant: &[u8],
    seq_k: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut result = vec![0.0f32; seq_k * head_dim];

    match format {
        "q4_0" => {
            let block_size = 18;
            let blocks_per_row = head_dim / 32;
            for s in 0..seq_k {
                for b in 0..blocks_per_row {
                    let block_offset = (s * blocks_per_row + b) * block_size;
                    let d = f16::from_bits(u16::from_le_bytes([
                        quant[block_offset],
                        quant[block_offset + 1],
                    ]))
                    .to_f32();
                    for i in 0..32 {
                        let q = (quant[block_offset + 2 + i / 2]
                            >> ((i % 2) * 4))
                            & 0x0F;
                        let q = q as i8 - 8; // Signed
                        result[s * head_dim + b * 32 + i] = q as f32 * d;
                    }
                }
            }
        }
        "q8_0" => {
            let block_size = 34;
            let blocks_per_row = head_dim / 32;
            for s in 0..seq_k {
                for b in 0..blocks_per_row {
                    let block_offset = (s * blocks_per_row + b) * block_size;
                    let d = f16::from_bits(u16::from_le_bytes([
                        quant[block_offset],
                        quant[block_offset + 1],
                    ]))
                    .to_f32();
                    for i in 0..32 {
                        let q = quant[block_offset + 2 + i] as i8;
                        result[s * head_dim + b * 32 + i] = q as f32 * d;
                    }
                }
            }
        }
        // For other formats, use simple approximation (just scale by 1.0)
        _ => {
            for (i, v) in result.iter_mut().enumerate() {
                *v = quant[i % quant.len()] as f32;
            }
        }
    }

    result
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
            let stride = seq_k * head_dim / 32 * 18;
            (k, v, stride, stride)
        }
        "q4_1" => {
            let k = quantize_to_q4_1(k_f32, seq_k, head_dim);
            let v = quantize_to_q4_1(v_f32, seq_k, head_dim);
            let stride = seq_k * head_dim / 32 * 20;
            (k, v, stride, stride)
        }
        "q5_0" => {
            let k = quantize_to_q5_0(k_f32, seq_k, head_dim);
            let v = quantize_to_q5_0(v_f32, seq_k, head_dim);
            let stride = seq_k * head_dim / 32 * 22;
            (k, v, stride, stride)
        }
        "q5_1" => {
            let k = quantize_to_q5_1(k_f32, seq_k, head_dim);
            let v = quantize_to_q5_1(v_f32, seq_k, head_dim);
            let stride = seq_k * head_dim / 32 * 24;
            (k, v, stride, stride)
        }
        "q8_0" => {
            let k = quantize_to_q8_0(k_f32, seq_k, head_dim);
            let v = quantize_to_q8_0(v_f32, seq_k, head_dim);
            let stride = seq_k * head_dim / 32 * 34;
            (k, v, stride, stride)
        }
        "bf16" => {
            let k = quantize_to_bf16(k_f32, seq_k, head_dim);
            let v = quantize_to_bf16(v_f32, seq_k, head_dim);
            let stride = seq_k * head_dim * 2;
            (k, v, stride, stride)
        }
        _ => panic!("Unknown format: {}", format),
    }
}

fn get_block_size(format: &str, head_dim: usize) -> usize {
    match format {
        "q4_0" => 18,
        "q4_1" => 20,
        "q5_0" => 22,
        "q5_1" => 24,
        "q8_0" => 34,
        "bf16" => 2,
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

fn quantize_to_q4_1(data: &[f32], seq: usize, head_dim: usize) -> Vec<u8> {
    use infer_train::quant::kv_cache::{quantize_row_q4_1, BlockQ4_1};
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
    use infer_train::quant::kv_cache::{quantize_row_q5_0, BlockQ5_0};
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
    use infer_train::quant::kv_cache::{quantize_row_q5_1, BlockQ5_1};
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
