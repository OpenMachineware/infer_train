// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Q4_K and Q6_K GEMM with various sizes

use infer_train::quant::types::{BlockQ4K, BlockQ6K};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn test_q4k_gemm(ctx: &MetalContext, m: usize, n: usize, k: usize) {
    let num_blocks = (k * m) / 256;
    let weights: Vec<BlockQ4K> = (0..num_blocks)
        .map(|i| {
            let mut block = BlockQ4K {
                d: half::f16::from_f32(0.01).to_bits(),
                dmin: half::f16::from_f32(0.001).to_bits(),
                scales: [0u8; 12],
                qs: [0u8; 128],
            };
            for j in 0..12 {
                block.scales[j] = ((i + j) % 12 + 1) as u8;
            }
            for j in 0..128 {
                block.qs[j] = ((i + j) % 256) as u8;
            }
            block
        })
        .collect();

    let mut input = vec![0.01f32; n * k];
    for i in 0..n {
        for j in 0..k {
            input[i * k + j] = 0.001 * (i + j) as f32;
        }
    }

    println!("Testing Q4_K GEMM: M={}, N={}, K={}", m, n, k);
    let result = ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input);

    match result {
        Ok(output) => {
            let num_nan = output.iter().filter(|&&v| v.is_nan()).count();
            let num_inf = output.iter().filter(|&&v| v.is_infinite()).count();
            let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
            println!(
                "  Stats: nan={}, inf={}, min={:.4}, max={:.4}",
                num_nan, num_inf, min, max
            );
            if num_nan > 0 {
                println!("  FAILED");
            } else {
                println!("  SUCCESS");
            }
        }
        Err(e) => {
            println!("  ERROR: {}", e);
        }
    }
}

fn test_q6k_gemm(ctx: &MetalContext, m: usize, n: usize, k: usize) {
    let num_blocks = (k * m) / 256;
    let weights: Vec<BlockQ6K> = (0..num_blocks)
        .map(|i| {
            let mut block = BlockQ6K {
                ql: [0u8; 128],
                qh: [0u8; 64],
                scales: [0i8; 16],
                d: half::f16::from_f32(0.01).to_bits(),
            };
            for j in 0..16 {
                block.scales[j] = ((i + j) % 8 + 1) as i8;
            }
            for j in 0..128 {
                block.ql[j] = ((i + j) % 256) as u8;
            }
            for j in 0..64 {
                block.qh[j] = ((i + j) % 64) as u8;
            }
            block
        })
        .collect();

    let mut input = vec![0.01f32; n * k];
    for i in 0..n {
        for j in 0..k {
            input[i * k + j] = 0.001 * (i + j) as f32;
        }
    }

    println!("Testing Q6_K GEMM: M={}, N={}, K={}", m, n, k);
    let result = ctx.gemm_template_q6_k_f32(m, n, k, &weights, &input);

    match result {
        Ok(output) => {
            let num_nan = output.iter().filter(|&&v| v.is_nan()).count();
            let num_inf = output.iter().filter(|&&v| v.is_infinite()).count();
            let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
            println!(
                "  Stats: nan={}, inf={}, min={:.4}, max={:.4}",
                num_nan, num_inf, min, max
            );
            if num_nan > 0 {
                println!("  FAILED");
            } else {
                println!("  SUCCESS");
            }
        }
        Err(e) => {
            println!("  ERROR: {}", e);
        }
    }
}

fn main() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    // Test various K sizes
    println!("\n=== Q4_K GEMM Tests ===\n");
    test_q4k_gemm(&ctx, 64, 4, 256); // 1 block
    test_q4k_gemm(&ctx, 5120, 7, 4096); // Attention output projection size
    test_q4k_gemm(&ctx, 5120, 7, 5120); // Q projection size

    println!("\n=== Q6_K GEMM Tests ===\n");
    test_q6k_gemm(&ctx, 64, 4, 256); // 1 block
    test_q6k_gemm(&ctx, 5120, 7, 4096); // Attention output projection size
    test_q6k_gemm(&ctx, 5120, 7, 32768); // FFN down projection size
}
