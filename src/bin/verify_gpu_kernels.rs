// Verify GPU kernels against llama.cpp baselines
// Run: cargo run --release --bin verify_gpu_kernels

use infer_train::quant::types::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;
use std::time::{Duration, Instant};

fn generate_q2_k_weights(m: usize, k: usize) -> Vec<BlockQ2K> {
    let nb = k / 256;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        for j in 0..16 {
            scales[j] = ((i + j + 1) | ((i + j) << 4)) as u8;
        }
        for j in 0..64 {
            qs[j] = ((i * 23 + j * 7) % 256) as u8;
        }
        weights.push(BlockQ2K {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qs,
        });
    }
    weights
}

fn generate_q3_k_weights(m: usize, k: usize) -> Vec<BlockQ3K> {
    let nb = k / 256;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut hmask = [0u8; 32];
        let mut qs = [0u8; 64];
        let mut scales = [0u8; 12];
        for j in 0..32 {
            hmask[j] = if j % 2 == 0 { 0xFF } else { 0x00 };
        }
        for j in 0..64 {
            qs[j] = ((i * 17 + j * 4) % 256) as u8;
        }
        for j in 0..12 {
            scales[j] = (40 + (j % 8)) as u8;
        }
        weights.push(BlockQ3K {
            d: f16::from_f32(1.0).to_bits(),
            hmask,
            qs,
            scales,
        });
    }
    weights
}

fn generate_q4_k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut qs = [0u8; 128];
        let mut scales = [0u8; 12];
        for j in 0..128 {
            qs[j] = ((i * 17 + j * 13) % 256) as u8;
        }
        for j in 0..12 {
            scales[j] = ((i + j) % 64) as u8;
        }
        weights.push(BlockQ4K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qs,
        });
    }
    weights
}

fn generate_q5_k_weights(m: usize, k: usize) -> Vec<BlockQ5K> {
    let nb = k / 256;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut qs = [0u8; 128];
        let mut qh = [0u8; 32];
        let mut scales = [0u8; 12];
        for j in 0..128 {
            qs[j] = ((i * 17 + j * 13) % 256) as u8;
        }
        for j in 0..32 {
            qh[j] = ((i * 23 + j * 7) % 256) as u8;
        }
        for j in 0..12 {
            scales[j] = ((i + j) % 64) as u8;
        }
        weights.push(BlockQ5K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qh,
            qs,
        });
    }
    weights
}

fn generate_q6_k_weights(m: usize, k: usize) -> Vec<BlockQ6K> {
    let nb = k / 256;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        for j in 0..128 {
            ql[j] = ((i * 17 + j * 13) % 256) as u8;
        }
        for j in 0..64 {
            qh[j] = ((i * 23 + j * 7) % 256) as u8;
        }
        for j in 0..16 {
            scales[j] = (40 + (j % 8)) as i8;
        }
        weights.push(BlockQ6K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            ql,
            qh,
            scales,
        });
    }
    weights
}

fn generate_f32_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect()
}

fn calculate_gflops(m: usize, n: usize, k: usize, time_us: f64) -> f64 {
    let ops = 2.0 * m as f64 * n as f64 * k as f64;
    let time_s = time_us / 1_000_000.0;
    if time_s > 0.0 { ops / time_s / 1_000_000_000.0 } else { 0.0 }
}

fn benchmark_iterations<F>(f: F, iterations: usize) -> Duration
where
    F: Fn(),
{
    let mut total = Duration::new(0, 0);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        total += start.elapsed();
    }
    total
}

fn main() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         GPU Kernel Verification vs llama.cpp Baseline        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // MV Performance (Decode Path) - K=4096
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ MV Performance (Decode Path, K=4096)                        │");
    println!("├───────┬──────────┬──────────┬──────────┬──────────┬─────────┤");
    println!("│ M     │ Our (µs) │ llama.cpp│ Ratio    │ GFLOPS   │ Status  │");
    println!("├───────┼──────────┼──────────┼──────────┼──────────┼─────────┤");

    let k = 4096;
    let m_sizes = [256, 1024, 2048, 8192];

    // llama.cpp baseline from memory/rust-metal-mv-progress.md
    let llama_mv_q4k: [(usize, f64); 4] = [
        (256, 358.0),
        (1024, 368.0),
        (2048, 411.0),
        (8192, 638.0),
    ];

    for (idx, m) in m_sizes.iter().enumerate() {
        let weights = generate_q4_k_weights(*m, k);
        let input = generate_f32_input(k);

        // Warmup
        let _ = ctx.mv_q4_k_f32(*m, k, &weights, &input).unwrap();

        // Benchmark
        let iterations = 20;
        let total_time = benchmark_iterations(|| {
            let _ = ctx.mv_q4_k_f32(*m, k, &weights, &input).unwrap();
        }, iterations);
        let our_time = total_time.as_micros() as f64 / iterations as f64;

        let llama_time = llama_mv_q4k[idx].1;
        let ratio = llama_time / our_time;
        let gflops = calculate_gflops(*m, 1, k, our_time);
        let status = if ratio >= 1.0 { "✓ PASS" } else { "✗ FAIL" };

        println!("│ {:5} │ {:8.2} │ {:8.2} │ {:8.2}x │ {:8.2} │ {} │",
                 m, our_time, llama_time, ratio, gflops, status);
    }
    println!("└───────┴──────────┴──────────┴──────────┴──────────┴─────────┘\n");

    // GEMM Performance (Batch Inference) - M=4096, K=4096
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ GEMM Performance (Batch Inference, M=4096, K=4096)         │");
    println!("├───────┬──────────┬──────────┬──────────┬──────────┬─────────┤");
    println!("│ N     │ Our (µs) │ llama.cpp│ Ratio    │ GFLOPS   │ Status  │");
    println!("├───────┼──────────┼──────────┼──────────┼──────────┼─────────┤");

    let m = 4096;
    let n_sizes = [1, 4, 16, 64, 128];

    for n in &n_sizes {
        let weights = generate_q4_k_weights(m, k);
        let input = generate_f32_input(k * *n);

        // Warmup
        let _ = ctx.gemm_q4_k_f32(m, *n, k, &weights, &input).unwrap();

        // Benchmark
        let iterations = 10;
        let total_time = benchmark_iterations(|| {
            let _ = ctx.gemm_q4_k_f32(m, *n, k, &weights, &input).unwrap();
        }, iterations);
        let our_time = total_time.as_micros() as f64 / iterations as f64;

        // Estimate llama.cpp time (based on ~200-260 GFLOPS for large N)
        let llama_gflops = 220.0; // Average from memory
        let llama_ops = 2.0 * m as f64 * *n as f64 * k as f64;
        let llama_time = llama_ops / llama_gflops / 1_000_000_000.0 * 1_000_000.0;

        let gflops = calculate_gflops(m, *n, k, our_time);
        let ratio = llama_time / our_time;
        let status = if gflops >= llama_gflops * 0.95 { "✓ PASS" } else { "✗ FAIL" };

        println!("│ {:5} │ {:8.2} │ {:8.2} │ {:8.2}x │ {:8.2} │ {} │",
                 n, our_time, llama_time, ratio, gflops, status);
    }
    println!("└───────┴──────────┴──────────┴──────────┴──────────┴─────────┘\n");

    // All K-Quant Formats
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ All K-Quant Formats (GEMM M=4096, N=128, K=4096)           │");
    println!("├────────┬──────────┬──────────┬──────────┬──────────┬────────┤");
    println!("│ Format │ Our (µs) │ GFLOPS   │ llama.cpp│ Ratio    │ Status │");
    println!("├────────┼──────────┼──────────┼──────────┼──────────┼────────┤");

    let n = 128;
    let formats: [(&str, fn(usize, usize) -> Vec<BlockQ4K>); 5] = [
        ("Q2_K", generate_q4_k_weights as fn(usize, usize) -> Vec<BlockQ4K>),
        ("Q3_K", generate_q4_k_weights as fn(usize, usize) -> Vec<BlockQ4K>),
        ("Q4_K", generate_q4_k_weights),
        ("Q5_K", generate_q4_k_weights as fn(usize, usize) -> Vec<BlockQ4K>),
        ("Q6_K", generate_q4_k_weights as fn(usize, usize) -> Vec<BlockQ4K>),
    ];

    let llama_gflops_baseline = 220.0;

    for (format_name, _) in &formats {
        let weights: Vec<u8> = vec![0; m * k / 8]; // Placeholder
        let input = generate_f32_input(k * n);

        let (our_time, gflops) = match *format_name {
            "Q2_K" => {
                let weights = generate_q2_k_weights(m, k);
                let _ = ctx.gemm_q2_k_f32(m, n, k, &weights, &input).unwrap();
                let total = benchmark_iterations(|| {
                    let _ = ctx.gemm_q2_k_f32(m, n, k, &weights, &input).unwrap();
                }, 10);
                let t = total.as_micros() as f64 / 10.0;
                (t, calculate_gflops(m, n, k, t))
            }
            "Q3_K" => {
                let weights = generate_q3_k_weights(m, k);
                let _ = ctx.gemm_q3_k_f32(m, n, k, &weights, &input).unwrap();
                let total = benchmark_iterations(|| {
                    let _ = ctx.gemm_q3_k_f32(m, n, k, &weights, &input).unwrap();
                }, 10);
                let t = total.as_micros() as f64 / 10.0;
                (t, calculate_gflops(m, n, k, t))
            }
            "Q4_K" => {
                let weights = generate_q4_k_weights(m, k);
                let _ = ctx.gemm_q4_k_f32(m, n, k, &weights, &input).unwrap();
                let total = benchmark_iterations(|| {
                    let _ = ctx.gemm_q4_k_f32(m, n, k, &weights, &input).unwrap();
                }, 10);
                let t = total.as_micros() as f64 / 10.0;
                (t, calculate_gflops(m, n, k, t))
            }
            "Q5_K" => {
                let weights = generate_q5_k_weights(m, k);
                let _ = ctx.gemm_q5_k_f32(m, n, k, &weights, &input).unwrap();
                let total = benchmark_iterations(|| {
                    let _ = ctx.gemm_q5_k_f32(m, n, k, &weights, &input).unwrap();
                }, 10);
                let t = total.as_micros() as f64 / 10.0;
                (t, calculate_gflops(m, n, k, t))
            }
            "Q6_K" => {
                let weights = generate_q6_k_weights(m, k);
                let _ = ctx.gemm_q6_k_f32(m, n, k, &weights, &input).unwrap();
                let total = benchmark_iterations(|| {
                    let _ = ctx.gemm_q6_k_f32(m, n, k, &weights, &input).unwrap();
                }, 10);
                let t = total.as_micros() as f64 / 10.0;
                (t, calculate_gflops(m, n, k, t))
            }
            _ => (0.0, 0.0),
        };

        let ratio = gflops / llama_gflops_baseline;
        let status = if ratio >= 0.95 { "✓ PASS" } else { "✗ FAIL" };

        println!("│ {:6} │ {:8.2} │ {:8.2} │ {:8.2} │ {:8.2}x │ {} │",
                 format_name, our_time, gflops, llama_gflops_baseline, ratio, status);
    }
    println!("└────────┴──────────┴──────────┴──────────┴──────────┴────────┘\n");

    // Small K Performance
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Small K Performance (GEMM M=256, N=16)                      │");
    println!("├────────┬──────────┬──────────┬──────────┬──────────┬────────┤");
    println!("│ K      │ Our (µs) │ GFLOPS   │ Expected │ Ratio    │ Status │");
    println!("├────────┼──────────┼──────────┼──────────┼──────────┼────────┤");

    let m_small = 256;
    let n_small = 16;
    let k_small_sizes = [256, 512, 1024, 2048, 4096];

    for k_small in &k_small_sizes {
        let weights = generate_q4_k_weights(m_small, *k_small);
        let input = generate_f32_input(*k_small * n_small);

        let _ = ctx.gemm_q4_k_f32(m_small, n_small, *k_small, &weights, &input).unwrap();

        let total = benchmark_iterations(|| {
            let _ = ctx.gemm_q4_k_f32(m_small, n_small, *k_small, &weights, &input).unwrap();
        }, 20);
        let our_time = total.as_micros() as f64 / 20.0;
        let gflops = calculate_gflops(m_small, n_small, *k_small, our_time);

        // Expected: should scale with K (memory bandwidth limited for small K)
        let expected_gflops = 30.0 * (*k_small as f64 / 256.0).min(4.0);
        let ratio = gflops / expected_gflops;
        let status = if ratio >= 0.5 { "✓ PASS" } else { "✗ FAIL" };

        println!("│ {:6} │ {:8.2} │ {:8.2} │ {:8.2} │ {:8.2}x │ {} │",
                 k_small, our_time, gflops, expected_gflops, ratio, status);
    }
    println!("└────────┴──────────┴──────────┴──────────┴──────────┴────────┘\n");

    println!("Verification complete.\n");
}
