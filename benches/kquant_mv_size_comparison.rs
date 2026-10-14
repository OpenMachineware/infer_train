// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// K-quant MV performance comparison across different sizes
// Alternating benchmark: llama.cpp vs self-written (via MetalContext)

use bytemuck::Zeroable;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use std::time::Instant;

fn generate_weights_q2k(m: usize, k: usize) -> Vec<BlockQ2K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut block = BlockQ2K::zeroed();
            // Set scales to valid values (1-16)
            for j in 0..16 {
                block.scales[j] = ((i + j) % 16 + 1) as u8;
            }
            // Set qs to pattern
            for j in 0..64 {
                block.qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            // d = 1.0 in f16
            block.d = 0x3C00;
            // dmin = 0.0
            block.dmin = 0;
            block
        })
        .collect()
}

fn generate_weights_q3k(m: usize, k: usize) -> Vec<BlockQ3K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut block = BlockQ3K::zeroed();
            // hmask pattern
            for j in 0..32 {
                block.hmask[j] = ((i + j) % 256) as u8;
            }
            // qs pattern
            for j in 0..64 {
                block.qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            // scales pattern
            for j in 0..12 {
                block.scales[j] = ((i + j) % 256) as u8;
            }
            // d = 1.0
            block.d = 0x3C00;
            block
        })
        .collect()
}

fn generate_weights_q4k(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut block = BlockQ4K::zeroed();
            block.d = 0x3C00;
            block.dmin = 0;
            for j in 0..12 {
                block.scales[j] = ((i + j) % 16 + 1) as u8;
            }
            for j in 0..128 {
                block.qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            block
        })
        .collect()
}

fn generate_weights_q5k(m: usize, k: usize) -> Vec<BlockQ5K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut block = BlockQ5K::zeroed();
            block.d = 0x3C00;
            block.dmin = 0;
            for j in 0..12 {
                block.scales[j] = ((i + j) % 16 + 1) as u8;
            }
            for j in 0..32 {
                block.qh[j] = ((i + j) % 256) as u8;
            }
            for j in 0..128 {
                block.qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            block
        })
        .collect()
}

fn generate_weights_q6k(m: usize, k: usize) -> Vec<BlockQ6K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut block = BlockQ6K::zeroed();
            for j in 0..128 {
                block.ql[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..64 {
                block.qh[j] = ((i + j) % 256) as u8;
            }
            for j in 0..16 {
                block.scales[j] = ((i + j) % 64 + 1) as i8;
            }
            block.d = 0x3C00;
            block
        })
        .collect()
}

fn generate_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

fn main() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    // Test sizes: small to large (K must be multiple of 256 for K-quant)
    let sizes = [
        (256, 256),
        (256, 1024),
        (256, 4096),
        (1024, 256),
        (1024, 1024),
        (1024, 4096),
        (4096, 256),
        (4096, 1024),
        (4096, 4096),
    ];

    let alternating_runs = 50;
    let iterations_per_run = 50;

    println!(
        "K-quant MV Performance Comparison (Self-written via MetalContext)"
    );
    println!(
        "================================================================="
    );
    println!(
        "Each test: {} runs x {} iterations\n",
        alternating_runs, iterations_per_run
    );

    // Test Q4_K
    println!("\n## Q4_K (144 bytes/block)");
    println!("M\tK\tTime(us)\tGFLOPS");

    for (m, k) in sizes {
        let weights = generate_weights_q4k(m, k);
        let input = generate_input(k);

        // Warmup
        for _ in 0..20 {
            let _ = ctx.mv_q4_k_f32(m, k, &weights, &input);
        }

        // Benchmark
        let mut times = Vec::with_capacity(alternating_runs);
        for _ in 0..alternating_runs {
            let mut total_time = 0u128;
            for _ in 0..iterations_per_run {
                let start = Instant::now();
                let _ = ctx.mv_q4_k_f32(m, k, &weights, &input);
                total_time += start.elapsed().as_micros();
            }
            times.push(total_time as f64 / iterations_per_run as f64);
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[alternating_runs / 2];
        let flops = 2.0 * m as f64 * k as f64;
        let gflops = flops / median / 1000.0;

        println!("{}\t{}\t{:.2}\t\t{:.1}", m, k, median, gflops);
    }

    // Test Q2_K
    println!("\n## Q2_K (84 bytes/block)");
    println!("M\tK\tTime(us)\tGFLOPS");

    for (m, k) in sizes {
        let weights = generate_weights_q2k(m, k);
        let input = generate_input(k);

        for _ in 0..20 {
            let _ = ctx.mv_q2_k_f32(m, k, &weights, &input);
        }

        let mut times = Vec::with_capacity(alternating_runs);
        for _ in 0..alternating_runs {
            let mut total_time = 0u128;
            for _ in 0..iterations_per_run {
                let start = Instant::now();
                let _ = ctx.mv_q2_k_f32(m, k, &weights, &input);
                total_time += start.elapsed().as_micros();
            }
            times.push(total_time as f64 / iterations_per_run as f64);
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[alternating_runs / 2];
        let gflops = 2.0 * m as f64 * k as f64 / median / 1000.0;

        println!("{}\t{}\t{:.2}\t\t{:.1}", m, k, median, gflops);
    }

    // Test Q3_K
    println!("\n## Q3_K (110 bytes/block)");
    println!("M\tK\tTime(us)\tGFLOPS");

    for (m, k) in sizes {
        let weights = generate_weights_q3k(m, k);
        let input = generate_input(k);

        for _ in 0..20 {
            let _ = ctx.mv_q3_k_f32(m, k, &weights, &input);
        }

        let mut times = Vec::with_capacity(alternating_runs);
        for _ in 0..alternating_runs {
            let mut total_time = 0u128;
            for _ in 0..iterations_per_run {
                let start = Instant::now();
                let _ = ctx.mv_q3_k_f32(m, k, &weights, &input);
                total_time += start.elapsed().as_micros();
            }
            times.push(total_time as f64 / iterations_per_run as f64);
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[alternating_runs / 2];
        let gflops = 2.0 * m as f64 * k as f64 / median / 1000.0;

        println!("{}\t{}\t{:.2}\t\t{:.1}", m, k, median, gflops);
    }

    // Test Q5_K
    println!("\n## Q5_K (176 bytes/block)");
    println!("M\tK\tTime(us)\tGFLOPS");

    for (m, k) in sizes {
        let weights = generate_weights_q5k(m, k);
        let input = generate_input(k);

        for _ in 0..20 {
            let _ = ctx.mv_q5_k_f32(m, k, &weights, &input);
        }

        let mut times = Vec::with_capacity(alternating_runs);
        for _ in 0..alternating_runs {
            let mut total_time = 0u128;
            for _ in 0..iterations_per_run {
                let start = Instant::now();
                let _ = ctx.mv_q5_k_f32(m, k, &weights, &input);
                total_time += start.elapsed().as_micros();
            }
            times.push(total_time as f64 / iterations_per_run as f64);
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[alternating_runs / 2];
        let gflops = 2.0 * m as f64 * k as f64 / median / 1000.0;

        println!("{}\t{}\t{:.2}\t\t{:.1}", m, k, median, gflops);
    }

    // Test Q6_K
    println!("\n## Q6_K (210 bytes/block)");
    println!("M\tK\tTime(us)\tGFLOPS");

    for (m, k) in sizes {
        let weights = generate_weights_q6k(m, k);
        let input = generate_input(k);

        for _ in 0..20 {
            let _ = ctx.mv_q6_k_f32(m, k, &weights, &input);
        }

        let mut times = Vec::with_capacity(alternating_runs);
        for _ in 0..alternating_runs {
            let mut total_time = 0u128;
            for _ in 0..iterations_per_run {
                let start = Instant::now();
                let _ = ctx.mv_q6_k_f32(m, k, &weights, &input);
                total_time += start.elapsed().as_micros();
            }
            times.push(total_time as f64 / iterations_per_run as f64);
        }

        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[alternating_runs / 2];
        let gflops = 2.0 * m as f64 * k as f64 / median / 1000.0;

        println!("{}\t{}\t{:.2}\t\t{:.1}", m, k, median, gflops);
    }
}
