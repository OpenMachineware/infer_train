// SPDX-FileCopyrightText: 2026 Jia Liu
// SPDX-License-Identifier: AGPL-3.0-only

// Test correctness of new vectorized decode kernel

use metal::*;

const NUM_HEADS: usize = 32;
const HEAD_DIM: usize = 128;

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let source = include_str!("../src/ops/metal/flash_attn_quant.metal");
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    let library = device.new_library_with_source(source, &options).unwrap();

    println!("=== Flash Decode Vec Kernel Correctness Test ===\n");

    let seq_ks = [64, 128, 256, 512];

    for seq_k in &seq_ks {
        test_correctness(
            &device,
            &queue,
            &library,
            "flash_decode_grid8x4_f16",
            *seq_k,
            "F16",
        );
    }
}

fn test_correctness(
    device: &Device,
    queue: &CommandQueue,
    library: &Library,
    kernel_name: &str,
    seq_k: usize,
    fmt_name: &str,
) {
    let pipeline = library.get_function(kernel_name, None).unwrap();
    let pso =
        device.new_compute_pipeline_state_with_function(&pipeline).unwrap();

    // Create test data in half format (F16)
    use half::f16;
    let q_data: Vec<f16> = (0..NUM_HEADS * HEAD_DIM)
        .map(|i| f16::from_f32((i % 10) as f32 * 0.1))
        .collect();
    let k_data: Vec<f16> = (0..seq_k * HEAD_DIM)
        .map(|i| f16::from_f32((i % 20) as f32 * 0.05))
        .collect();
    let v_data: Vec<f16> = (0..seq_k * HEAD_DIM)
        .map(|i| f16::from_f32((i % 15) as f32 * 0.03))
        .collect();

    let q_buf = device.new_buffer_with_data(
        q_data.as_ptr() as *const _,
        (q_data.len() * 2) as u64, // half is 2 bytes
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
        (NUM_HEADS * HEAD_DIM * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Reference implementation (CPU softmax)
    let mut ref_output = vec![0.0f32; NUM_HEADS * HEAD_DIM];
    for h in 0..NUM_HEADS {
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

        // Compute scores
        let mut scores = vec![0.0f32; seq_k];
        for s in 0..seq_k {
            let mut score = 0.0f32;
            for d in 0..HEAD_DIM {
                score += q_data[h * HEAD_DIM + d].to_f32()
                    * k_data[s * HEAD_DIM + d].to_f32();
            }
            scores[s] = score * scale;
        }

        // Softmax
        let max_score =
            scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();

        // Weighted sum
        for s in 0..seq_k {
            let weight = (scores[s] - max_score).exp() / sum;
            for d in 0..HEAD_DIM {
                ref_output[h * HEAD_DIM + d] +=
                    weight * v_data[s * HEAD_DIM + d].to_f32();
            }
        }
    }

    // Run GPU kernel
    let args = FlashAttnArgs {
        batch: 1,
        heads: NUM_HEADS as u32,
        seq_q: 1,
        seq_k: seq_k as u32,
        head_dim: HEAD_DIM as u32,
        scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        q_stride_b: (NUM_HEADS * HEAD_DIM) as u32, // element stride
        q_stride_h: HEAD_DIM as u32,               // element stride
        q_stride_s: HEAD_DIM as u32,               // element stride
        k_stride_b: (seq_k * HEAD_DIM * 4) as u32, // byte stride
        k_stride_h: (seq_k * HEAD_DIM * 4) as u32, // byte stride
        k_stride_s: (HEAD_DIM * 4) as u32,         // byte stride
        v_stride_b: (seq_k * HEAD_DIM * 4) as u32, // byte stride
        v_stride_h: (seq_k * HEAD_DIM * 4) as u32, // byte stride
        v_stride_s: (HEAD_DIM * 4) as u32,         // byte stride
        o_stride_b: (NUM_HEADS * HEAD_DIM) as u32, // element stride
        o_stride_h: HEAD_DIM as u32,               // element stride
        o_stride_s: HEAD_DIM as u32,               // element stride
    };

    let kernel_name = "flash_decode_grid8x4_f16";

    let args_bytes = unsafe {
        std::slice::from_raw_parts(
            &args as *const _ as *const u8,
            std::mem::size_of::<FlashAttnArgs>(),
        )
    };

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    enc.set_bytes(0, args_bytes.len() as u64, args_bytes.as_ptr() as *const _);
    enc.set_buffer(1, Some(&q_buf), 0);
    enc.set_buffer(2, Some(&k_buf), 0);
    enc.set_buffer(3, Some(&v_buf), 0);
    enc.set_buffer(4, Some(&out_buf), 0);
    enc.set_threadgroup_memory_length(0, 128 * 2 * 2);
    enc.dispatch_thread_groups(
        MTLSize::new(1, NUM_HEADS as u64, 1),
        MTLSize::new(32, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Compare outputs
    let gpu_output = unsafe {
        std::slice::from_raw_parts(
            out_buf.contents() as *const f32,
            NUM_HEADS * HEAD_DIM,
        )
    };

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for i in 0..NUM_HEADS * HEAD_DIM {
        let diff = (gpu_output[i] - ref_output[i]).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let avg_diff = sum_diff / (NUM_HEADS * HEAD_DIM) as f32;

    println!(
        "{} seq_k={}: max_diff={:.6e}, avg_diff={:.6e}",
        fmt_name, seq_k, max_diff, avg_diff
    );

    if max_diff < 1e-4 {
        println!("  ✓ PASS");
    } else {
        println!("  ✗ FAIL - differences too large");
        // Print some sample outputs for debugging
        println!("  Sample outputs (head 0, first 8 elements):");
        for i in 0..8 {
            println!(
                "    GPU: {:.6}, Ref: {:.6}, Diff: {:.6}",
                gpu_output[i],
                ref_output[i],
                (gpu_output[i] - ref_output[i]).abs()
            );
        }
    }
}

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
