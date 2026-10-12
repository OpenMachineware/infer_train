// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test dequantization correctness
use infer_train::quant::kv_cache::{BlockQ4_0, BlockQ8_0};
use metal::*;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("=== Dequantization Test ===\n");

    // Test Q4_0
    test_q4_0(&device, &queue);

    // Test Q8_0
    test_q8_0(&device, &queue);
}

fn test_q4_0(device: &Device, queue: &CommandQueue) {
    println!("Testing Q4_0 dequantization:");

    // Create a single Q4_0 block
    let block = BlockQ4_0 {
        d: 0x3C00u16,   // 1.0 in FP16
        qs: [0x88; 16], // All nibbles = 8 (which becomes 0 after -8 offset)
    };

    println!("  Block: d=1.0, qs=[0x88; 16]");
    println!("  Expected: 32 zeros (d * (8 - 8) = 0)");

    // CPU dequant
    let cpu_result = dequantize_q4_0_cpu(&block);
    println!("  CPU result [0..4]: {:?}", &cpu_result[0..4]);

    // GPU dequant via a simple kernel
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

kernel void test_dequant(device const block_q4_0 * block [[buffer(0)]],
                         device float * output [[buffer(1)]]) {
    for (short il = 0; il < 8; il++) {
        float4 reg;
        dequantize_q4_0_t4(block, il, reg);
        output[il * 4 + 0] = reg[0];
        output[il * 4 + 1] = reg[1];
        output[il * 4 + 2] = reg[2];
        output[il * 4 + 3] = reg[3];
    }
}
"#;

    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(source, &options)
        .expect("Compile failed");
    let kernel =
        library.get_function("test_dequant", None).expect("Kernel not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("Pipeline failed");

    let block_bytes = unsafe {
        std::slice::from_raw_parts(
            &block as *const _ as *const u8,
            std::mem::size_of::<BlockQ4_0>(),
        )
    };

    let block_buf = device.new_buffer(
        block_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(128, MTLResourceOptions::StorageModeShared);

    unsafe {
        std::ptr::copy_nonoverlapping(
            block_bytes.as_ptr(),
            block_buf.contents() as *mut u8,
            block_bytes.len(),
        );
    }

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&block_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 1, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_result = unsafe {
        std::slice::from_raw_parts(out_buf.contents() as *const f32, 32)
    };

    println!("  GPU result [0..4]: {:?}", &gpu_result[0..4]);

    let max_diff: f32 = cpu_result
        .iter()
        .zip(gpu_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("  Max diff: {:.6e}", max_diff);
    println!("  Status: {}\n", if max_diff < 1e-5 { "PASS" } else { "FAIL" });
}

fn test_q8_0(device: &Device, queue: &CommandQueue) {
    println!("Testing Q8_0 dequantization:");

    // Create a single Q8_0 block
    let block = BlockQ8_0 {
        d: 0x3C00u16, // 1.0 in FP16
        qs: [8i8; 32],
    };

    println!("  Block: d=1.0, qs=[8; 32]");
    println!("  Expected: 32 values of 8.0");

    // CPU dequant
    let cpu_result = dequantize_q8_0_cpu(&block);
    println!("  CPU result [0..4]: {:?}", &cpu_result[0..4]);

    // GPU dequant
    let source = r#"
#include <metal_stdlib>
using namespace metal;

#define QK 32

struct block_q8_0 {
    half d;
    int8_t qs[QK];
};

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;
    reg = (type4) (float4(qs[il]) * d);
}

kernel void test_dequant(device const block_q8_0 * block [[buffer(0)]],
                         device float * output [[buffer(1)]]) {
    for (short il = 0; il < 8; il++) {
        float4 reg;
        dequantize_q8_0_t4(block, il, reg);
        output[il * 4 + 0] = reg[0];
        output[il * 4 + 1] = reg[1];
        output[il * 4 + 2] = reg[2];
        output[il * 4 + 3] = reg[3];
    }
}
"#;

    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(source, &options)
        .expect("Compile failed");
    let kernel =
        library.get_function("test_dequant", None).expect("Kernel not found");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("Pipeline failed");

    let block_bytes = unsafe {
        std::slice::from_raw_parts(
            &block as *const _ as *const u8,
            std::mem::size_of::<BlockQ8_0>(),
        )
    };

    let block_buf = device.new_buffer(
        block_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(128, MTLResourceOptions::StorageModeShared);

    unsafe {
        std::ptr::copy_nonoverlapping(
            block_bytes.as_ptr(),
            block_buf.contents() as *mut u8,
            block_bytes.len(),
        );
    }

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&block_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: 1, height: 1, depth: 1 },
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let gpu_result = unsafe {
        std::slice::from_raw_parts(out_buf.contents() as *const f32, 32)
    };

    println!("  GPU result [0..4]: {:?}", &gpu_result[0..4]);

    let max_diff: f32 = cpu_result
        .iter()
        .zip(gpu_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("  Max diff: {:.6e}", max_diff);
    println!("  Status: {}\n", if max_diff < 1e-5 { "PASS" } else { "FAIL" });
}

fn dequantize_q4_0_cpu(block: &BlockQ4_0) -> [f32; 32] {
    let d = half::f16::from_bits(block.d).to_f32();
    let mut output = [0.0f32; 32];
    for i in 0..16 {
        let q = block.qs[i];
        // Low nibble
        output[i] = d * ((q & 0x0F) as f32 - 8.0);
        // High nibble
        output[i + 16] = d * ((q >> 4) as f32 - 8.0);
    }
    output
}

fn dequantize_q8_0_cpu(block: &BlockQ8_0) -> [f32; 32] {
    let d = half::f16::from_bits(block.d).to_f32();
    let mut output = [0.0f32; 32];
    for i in 0..32 {
        output[i] = d * block.qs[i] as f32;
    }
    output
}
