// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Simple Q*K test for quantized flash attention
use infer_train::quant::kv_cache::{quantize_row_q4_0, BlockQ4_0};
use metal::*;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("=== Simple Q*K Test ===\n");

    // Create Q (one row, head_dim=128)
    let head_dim = 128;
    let mut seed: u64 = 42;
    let q: Vec<f32> = (0..head_dim).map(|_| rng(&mut seed)).collect();

    // Create K (one row)
    let k: Vec<f32> = (0..head_dim).map(|_| rng(&mut seed)).collect();

    // Quantize K
    let blocks_per_row = head_dim / 32;
    let mut k_blocks: Vec<BlockQ4_0> = Vec::with_capacity(blocks_per_row);
    unsafe {
        k_blocks.set_len(blocks_per_row);
    }
    quantize_row_q4_0(&k, &mut k_blocks);

    // CPU reference
    let mut cpu_score = 0.0f32;
    for d in 0..head_dim {
        let k_deq = dequantize_q4_0_value(&k_blocks, d);
        cpu_score += q[d] * k_deq;
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    cpu_score *= scale;
    println!("CPU Q*K score: {:.6}", cpu_score);

    // GPU kernel
    let source = r#"
#include <metal_stdlib>
using namespace metal;

#define QK 32

struct block_q4_0 {
    half d;
    uint8_t qs[QK / 2];
};

template <typename type4>
void dequantize_q4_0_t4(device const block_q4_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = (il/4) ? (xb->d / 16.0h) : xb->d;
    const float d2 = d1 / 256.0f;
    const float md = -8.0h * xb->d;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}

struct Args {
    uint32_t head_dim;
    float scale;
    uint32_t k_stride_s;
};

kernel void test_qk(
    constant Args & args [[buffer(0)]],
    device const float * q [[buffer(1)]],
    device const char * k [[buffer(2)]],
    device float * score_out [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    constexpr ushort BLK_SIZE = 32;
    constexpr ushort BLK4 = BLK_SIZE / 4;

    // Dequantize K
    float k_deq[128];
    for (ushort b = 0; b < args.head_dim / BLK_SIZE; b++) {
        device const block_q4_0 * kb = (device const block_q4_0 *)(k + b * sizeof(block_q4_0));
        for (ushort il = 0; il < BLK4; il++) {
            float4 reg;
            dequantize_q4_0_t4(kb, il, reg);
            ushort d4 = b * BLK4 + il;
            k_deq[d4 * 4 + 0] = reg[0];
            k_deq[d4 * 4 + 1] = reg[1];
            k_deq[d4 * 4 + 2] = reg[2];
            k_deq[d4 * 4 + 3] = reg[3];
        }
    }

    // Q*K
    float score = 0.0f;
    for (ushort d = 0; d < args.head_dim; d++) {
        score += q[d] * k_deq[d];
    }
    score *= args.scale;

    *score_out = score;
}
"#;

    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(source, &options)
        .expect("Compile failed");
    let kernel =
        library.get_function("test_qk", None).expect("Kernel not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("Pipeline failed");

    let q_buf = device.new_buffer(
        (q.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let k_bytes = unsafe {
        std::slice::from_raw_parts(
            k_blocks.as_ptr() as *const u8,
            k_blocks.len() * std::mem::size_of::<BlockQ4_0>(),
        )
    };
    let k_buf = device.new_buffer(
        k_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let score_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    unsafe {
        std::ptr::copy_nonoverlapping(
            q.as_ptr(),
            q_buf.contents() as *mut f32,
            q.len(),
        );
        std::ptr::copy_nonoverlapping(
            k_bytes.as_ptr(),
            k_buf.contents() as *mut u8,
            k_bytes.len(),
        );
    }

    #[repr(C)]
    struct Args {
        head_dim: u32,
        scale: f32,
        k_stride_s: u32,
    }

    let args = Args {
        head_dim: head_dim as u32,
        scale,
        k_stride_s: (blocks_per_row * std::mem::size_of::<BlockQ4_0>()) as u32,
    };

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_bytes(
        0,
        std::mem::size_of::<Args>() as u64,
        &args as *const _ as *const _,
    );
    enc.set_buffer(1, Some(&q_buf), 0);
    enc.set_buffer(2, Some(&k_buf), 0);
    enc.set_buffer(3, Some(&score_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 1, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_score = unsafe { *(score_buf.contents() as *const f32) };
    println!("GPU Q*K score: {:.6}", gpu_score);
    println!("Diff: {:.6e}", (cpu_score - gpu_score).abs());
    println!(
        "Status: {}",
        if (cpu_score - gpu_score).abs() < 1e-3 { "PASS" } else { "FAIL" }
    );
}

fn rng(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn dequantize_q4_0_value(blocks: &[BlockQ4_0], idx: usize) -> f32 {
    let block_idx = idx / 32;
    let elem_idx = idx % 32;
    let block = &blocks[block_idx];
    let d = half::f16::from_bits(block.d).to_f32();

    // Match Metal's interleaved layout
    let il = elem_idx / 4;
    let sub_idx = elem_idx % 4;

    let qs_u16 = unsafe {
        std::slice::from_raw_parts(block.qs.as_ptr() as *const u16, 8)
    };

    let d1 = if il >= 4 { d / 16.0 } else { d };
    let d2 = d1 / 256.0;
    let md = -8.0 * d;
    let mask0: u16 = if il >= 4 { 0x00F0 } else { 0x000F };
    let mask1 = mask0 << 8;

    let i = if sub_idx < 2 { 0 } else { 1 };
    let q_val = qs_u16[2 * (il % 4) + i];

    if sub_idx % 2 == 0 {
        d1 * (q_val & mask0) as f32 + md
    } else {
        d2 * (q_val & mask1) as f32 + md
    }
}
