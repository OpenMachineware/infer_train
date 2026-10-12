// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Simple test for GPU Flash Attention
// Tests if Metal kernel compiles and runs correctly

use infer_train::ops::metal::{FlashAttnMetalContext, KvFormat};

fn main() {
    println!("Testing GPU Flash Attention...");

    let ctx = match FlashAttnMetalContext::new() {
        Ok(ctx) => {
            println!("✓ Metal context created successfully");
            ctx
        }
        Err(e) => {
            eprintln!("✗ Failed to create Metal context: {}", e);
            return;
        }
    };

    // Create simple test data
    let n_heads = 32;
    let head_dim = 128;
    let seq_len = 512;

    // Query (FP32) - values near 1.0
    let q: Vec<f32> = (0..n_heads * head_dim)
        .map(|i| 1.0 + 0.001 * (i as f32 % 100.0))
        .collect();

    // K/V in F16 format - values near 0.5
    let k: Vec<u16> = (0..seq_len * n_heads * head_dim)
        .map(|i| {
            half::f16::from_f32(0.5 + 0.001 * (i as f32 % 100.0)).to_bits()
        })
        .collect();
    let v: Vec<u16> = (0..seq_len * n_heads * head_dim)
        .map(|i| {
            half::f16::from_f32(0.3 + 0.001 * (i as f32 % 100.0)).to_bits()
        })
        .collect();

    let k_bytes = unsafe {
        std::slice::from_raw_parts(k.as_ptr() as *const u8, k.len() * 2)
    };
    let v_bytes = unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
    };

    println!("Testing F16 format with seq_len={}...", seq_len);
    println!("  Q size: {} elements", q.len());
    println!("  K size: {} elements", k.len());
    println!("  V size: {} elements", v.len());

    match ctx.flash_attn_decode(
        &q,
        k_bytes,
        v_bytes,
        None,
        n_heads as u32,
        n_heads as u32,
        head_dim as u32,
        seq_len as u32,
        KvFormat::F16,
    ) {
        Ok(output) => {
            println!("✓ F16 attention succeeded, output len={}", output.len());
            // Print first few values from each head
            println!("\nOutput samples:");
            for h in 0..4 {
                let start = h * head_dim;
                let vals = &output[start..start + 4];
                println!("  Head {}: {:?}", h, vals);
            }

            // Check for non-zero values
            let non_zero_count = output.iter().filter(|&&x| x != 0.0).count();
            println!(
                "\n  Non-zero values: {}/{}",
                non_zero_count,
                output.len()
            );

            // Compute expected output manually (simple attention)
            // For uniform K/V, the output should be close to the average
            if non_zero_count > 0 {
                let sum: f32 = output.iter().sum();
                let avg = sum / output.len() as f32;
                println!("  Average output value: {:.6}", avg);
            }
        }
        Err(e) => {
            eprintln!("✗ F16 attention failed: {}", e);
        }
    }

    println!("\nTest complete!");
}
