// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test KV cache GPU update kernel

use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::{MTLResourceOptions, NSRange};

fn main() {
    println!("=== KV Cache GPU Update Test ===\n");

    let ctx = MetalContext::new().expect("Failed to create Metal context");

    // Test parameters
    let n_kv_heads = 8;
    let head_dim = 64;
    let max_seq = 512;
    let pos = 0;

    // Create test input (F32)
    let input: Vec<f32> =
        (0..n_kv_heads * head_dim).map(|i| (i as f32 * 0.1).sin()).collect();

    // Create GPU buffers
    let input_buffer = ctx.device().new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = ctx.device().new_buffer(
        (n_kv_heads * max_seq * head_dim * 2) as u64, // F16 = 2 bytes
        MTLResourceOptions::StorageModeShared,
    );

    // Run KV update K kernel
    let cmd_buffer = ctx.queue().new_command_buffer();
    let encoder = cmd_buffer.new_compute_command_encoder();

    ctx.kv_update_k_encode(
        &encoder,
        &input_buffer,
        &output_buffer,
        pos,
        max_seq,
        n_kv_heads,
        head_dim,
    );

    encoder.end_encoding();
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();

    // Read back and verify
    let output_ptr = output_buffer.contents() as *const u16;
    let output_f16 = unsafe {
        std::slice::from_raw_parts(output_ptr, n_kv_heads * max_seq * head_dim)
    };

    println!("Input (first 4 elements): {:?}", &input[..4]);

    // Convert F16 back to F32 and compare
    for i in 0..4 {
        let f16_bits = output_f16[i];
        let f16_val = half::f16::from_bits(f16_bits);
        let f32_val = f16_val.to_f32();
        println!(
            "  Output[{}] = {} (F16 bits: 0x{:04X}, F16: {}, F32: {})",
            i, input[i], f16_bits, f16_val, f32_val
        );
        let diff = (input[i] - f32_val).abs();
        if diff > 0.001 {
            println!("    ✗ Mismatch: diff = {}", diff);
        } else {
            println!("    ✓ Match");
        }
    }

    println!("\n=== Test complete ===");
}
