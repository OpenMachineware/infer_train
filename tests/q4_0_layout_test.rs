// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Q4_0 layout and dequantization order
use infer_train::quant::kv_cache::BlockQ4_0;
use metal::*;

fn main() {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();

    println!("=== Q4_0 Layout Test ===\n");

    // Create a Q4_0 block with distinctive nibbles
    // Each byte has low nibble = low byte number, high nibble = high byte number
    // byte[0] = 0x10, byte[1] = 0x21, byte[2] = 0x32, ...
    let mut qs = [0u8; 16];
    for i in 0..16 {
        qs[i] = ((i % 16) as u8) | ((((i + 1) % 16) as u8) << 4);
    }

    let block = BlockQ4_0 {
        d: 0x3C00u16, // 1.0 in FP16
        qs,
    };

    println!("  Block: d=1.0");
    println!("  qs bytes: {:02x?}", qs);

    // CPU dequant - our understanding
    let cpu_result = dequantize_q4_0_cpu(&block);
    println!("\n  CPU dequant (low nibble first, then high nibble):");
    println!("  [0..8]: {:?}", &cpu_result[0..8]);
    println!("  [16..24]: {:?}", &cpu_result[16..24]);

    // GPU dequant
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

    println!("\n  GPU dequant (Metal template):");
    println!("  [0..8]: {:?}", &gpu_result[0..8]);
    println!("  [16..24]: {:?}", &gpu_result[16..24]);

    // Explain what Metal does
    println!("\n  Metal dequantize_q4_0_t4 logic:");
    println!("    il=0: processes qs[0] and qs[1] with mask 0x000F and 0x0F00");
    println!("    il=1: processes qs[2] and qs[3] with mask 0x000F and 0x0F00");
    println!("    il=2: processes qs[4] and qs[5] with mask 0x000F and 0x0F00");
    println!("    il=3: processes qs[6] and qs[7] with mask 0x000F and 0x0F00");
    println!("    il=4: processes qs[0] and qs[1] with mask 0x00F0 and 0xF000");
    println!("    il=5: processes qs[2] and qs[3] with mask 0x00F0 and 0xF000");
    println!("    ...");

    // Trace Metal manually
    println!("\n  Manual trace:");
    let d = 1.0f32;
    let md = -8.0f32 * d;
    let qs_u16 =
        unsafe { std::slice::from_raw_parts(qs.as_ptr() as *const u16, 8) };
    for il in 0..8 {
        let d1 = if il >= 4 { d / 16.0 } else { d };
        let d2 = d1 / 256.0;
        let mask0: u16 = if il >= 4 { 0x00F0 } else { 0x000F };
        let mask1 = mask0 << 8;

        for i in 0..2 {
            let q_val = qs_u16[2 * (il % 4) + i];
            let val0 = d1 * (q_val & mask0) as f32 + md;
            let val1 = d2 * (q_val & mask1) as f32 + md;
            let idx0 = il * 4 + 2 * i;
            let idx1 = il * 4 + 2 * i + 1;
            println!(
                "    il={}, i={}: qs_u16[{}]=0x{:04x}, mask0=0x{:04x}, mask1=0x{:04x} -> [{}]={:.3}, [{}]={:.3}",
                il,
                i,
                2 * (il % 4) + i,
                q_val,
                mask0,
                mask1,
                idx0,
                val0,
                idx1,
                val1
            );
        }
    }

    // Check alignment
    println!(
        "\n  Max diff: {:.6e}",
        cpu_result
            .iter()
            .zip(gpu_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    );
}

fn dequantize_q4_0_cpu(block: &BlockQ4_0) -> [f32; 32] {
    let d = half::f16::from_bits(block.d).to_f32();
    let mut output = [0.0f32; 32];
    for i in 0..16 {
        let q = block.qs[i];
        // Low nibble at position i
        output[i] = d * ((q & 0x0F) as f32 - 8.0);
        // High nibble at position i+16
        output[i + 16] = d * ((q >> 4) as f32 - 8.0);
    }
    output
}
