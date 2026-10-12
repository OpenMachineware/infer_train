// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark GPU Flash Attention - vec kernel (decode)
// Compares with llama.cpp expected performance

use infer_train::ops::metal::{FlashAttnMetalContext, KvFormat};
use std::time::Instant;

fn main() {
    println!("=== GPU Flash Attention Decode Benchmark ===\n");

    let ctx = match FlashAttnMetalContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to create Metal context: {}", e);
            return;
        }
    };

    let n_heads = 32;
    let n_kv_heads = 32;
    let head_dim = 128;

    // Test different sequence lengths
    let seq_lens = [128, 256, 512, 1024, 2048];
    let n_iterations = 20;

    println!("Vec Kernel (decode, single query):");
    println!("SeqLen | Time (μs) | GFLOPS | % vs llama.cpp*");
    println!("-------|-----------|--------|----------------");

    // llama.cpp expected performance (based on M1 measurements)
    // For decode attention, typical throughput is 20-30 GFLOPS on M1
    let llama_baseline_gflops = 25.0; // Approximate

    for &seq_len in &seq_lens {
        // Create test data with correct layout [n_heads, seq_len, head_dim]
        let q: Vec<f32> = (0..n_heads * head_dim)
            .map(|i| 0.5 + 0.01 * (i as f32 % 100.0))
            .collect();

        let mut k: Vec<f32> = vec![0.0; n_heads * seq_len * head_dim];
        let mut v: Vec<f32> = vec![0.0; n_heads * seq_len * head_dim];
        for h in 0..n_heads {
            for s in 0..seq_len {
                for d in 0..head_dim {
                    let idx = h * seq_len * head_dim + s * head_dim + d;
                    k[idx] =
                        0.3 + 0.01 * ((h * 1000 + s * 100 + d) as f32 % 100.0);
                    v[idx] =
                        0.2 + 0.01 * ((h * 1000 + s * 100 + d) as f32 % 100.0);
                }
            }
        }

        // Convert to F16
        let k_f16: Vec<u16> =
            k.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let v_f16: Vec<u16> =
            v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();

        let k_bytes = unsafe {
            std::slice::from_raw_parts(
                k_f16.as_ptr() as *const u8,
                k_f16.len() * 2,
            )
        };
        let v_bytes = unsafe {
            std::slice::from_raw_parts(
                v_f16.as_ptr() as *const u8,
                v_f16.len() * 2,
            )
        };

        // Warmup
        let _ = ctx.flash_attn_decode(
            &q,
            k_bytes,
            v_bytes,
            None,
            n_heads as u32,
            n_kv_heads as u32,
            head_dim as u32,
            seq_len as u32,
            KvFormat::F16,
        );

        // Benchmark
        let start = Instant::now();
        for _ in 0..n_iterations {
            let _ = ctx.flash_attn_decode(
                &q,
                k_bytes,
                v_bytes,
                None,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                seq_len as u32,
                KvFormat::F16,
            );
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / n_iterations as f64;

        // Compute GFLOPS
        let flops = 4.0 * n_heads as f64 * seq_len as f64 * head_dim as f64;
        let gflops = flops / (avg_us * 1000.0);

        let pct_vs_llama = (gflops / llama_baseline_gflops) * 100.0;

        println!(
            "{:>6} | {:>9.1} | {:>6.1} | {:>5.0}%",
            seq_len, avg_us, gflops, pct_vs_llama
        );
    }

    println!("\n* llama.cpp baseline: ~25 GFLOPS (estimated for M1 decode)");
    println!("\nDone.");
}
