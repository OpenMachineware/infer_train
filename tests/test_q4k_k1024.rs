// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Q4_K specifically at K=1024
use bytemuck::Zeroable;
use half::f16;
use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    let ctx = MetalContext::new().unwrap();

    // Test multiple sizes
    let test_cases = [(4096, 256), (4096, 1024), (4096, 4096)];

    for &(m, k) in &test_cases {
        let nb_per_row = k / 256;
        let total_blocks = m * nb_per_row;

        // Generate test weights
        let mut weights: Vec<BlockQ4K> = Vec::with_capacity(total_blocks);
        for i in 0..total_blocks {
            let mut block = BlockQ4K::zeroed();
            block.d = f16::to_bits(f16::from_f32(1.0));
            block.dmin = f16::to_bits(f16::from_f32(0.0));
            for j in 0..12 {
                block.scales[j] = ((i + j + 1) % 64) as u8;
            }
            for j in 0..128 {
                block.qs[j] = ((i * 128 + j) % 256) as u8;
            }
            weights.push(block);
        }

        // Generate test input
        let input: Vec<f32> = (0..k).map(|i| 0.1 * ((i % 10) as f32)).collect();

        // Run 20 times for stable timing
        let mut times = Vec::new();
        for _ in 0..3 {
            let _ = ctx.mv_q4_k_f32(m, k, &weights, &input);
        }
        for _ in 0..20 {
            let start = std::time::Instant::now();
            let _ = ctx.mv_q4_k_f32(m, k, &weights, &input);
            times.push(start.elapsed().as_nanos());
        }

        // Calculate GFLOPS
        let avg_ns: u128 = times.iter().sum::<u128>() / times.len() as u128;
        let gflops = 2.0 * m as f64 * k as f64 / (avg_ns as f64 / 1e9) / 1e9;

        // Expected baselines from llama.cpp
        let baseline = match (m, k) {
            (4096, 256) => 7.28,
            (4096, 1024) => 29.16,
            (4096, 4096) => 63.26,
            _ => 0.0,
        };

        let ratio = (gflops / baseline) * 100.0;
        println!(
            "M={}, K={}: {:.2} GFLOPS ({:.1}% of llama.cpp {:.2})",
            m, k, gflops, ratio, baseline
        );
    }
}
