#!/usr/bin/env cargo
//! Analyze why Load K/V and Softmax are slow

use std::time::{Duration, Instant};

fn main() {
    const HEAD_DIM: usize = 128;
    const KV_TILE_SZ: usize = 64;
    const N_ITERATIONS: usize = 1000;

    println!("=== Micro-benchmarking Load and Softmax ===\n");

    // Test data
    let k_data: Vec<f32> = (0..HEAD_DIM * KV_TILE_SZ * 10).map(|i| i as f32 * 0.001).collect();
    let mut k_tile: Vec<f32> = vec![0.0; HEAD_DIM * KV_TILE_SZ];
    let mut scores: Vec<f32> = (0..KV_TILE_SZ).map(|i| (i as f32 * 0.1).sin()).collect();

    // Test 1: Load K tile (naive scalar loop)
    println!("1. Load K tile (scalar loop):");
    let mut time_naive = Duration::new(0, 0);
    for _ in 0..N_ITERATIONS {
        let start = Instant::now();
        for tk in 0..KV_TILE_SZ {
            for d in 0..HEAD_DIM {
                k_tile[d * KV_TILE_SZ + tk] = k_data[tk * HEAD_DIM + d];
            }
        }
        time_naive += start.elapsed();
    }
    println!("  Time: {:?}", time_naive / N_ITERATIONS as u32);
    println!("  Bandwidth: {:.1} GB/s",
             (HEAD_DIM * KV_TILE_SZ * 4 * N_ITERATIONS) as f64 / time_naive.as_secs_f64() / 1e9);

    // Test 2: Load K tile (copy_from_slice)
    println!("\n2. Load K tile (copy_from_slice):");
    let mut time_copy = Duration::new(0, 0);
    for _ in 0..N_ITERATIONS {
        let start = Instant::now();
        for tk in 0..KV_TILE_SZ {
            let src_offset = tk * HEAD_DIM;
            let dst_offset = 0; // Wrong, but testing copy speed
            // This is not transposed, just testing copy speed
        }
        time_copy += start.elapsed();
    }

    // Test 3: Softmax (scalar)
    println!("\n3. Softmax (scalar loop):");
    let mut time_softmax = Duration::new(0, 0);
    for _ in 0..N_ITERATIONS {
        let start = Instant::now();

        // Find max
        let mut max_score = f32::NEG_INFINITY;
        for j in 0..KV_TILE_SZ {
            max_score = max_score.max(scores[j]);
        }

        // Exp and sum
        let mut sum_exp = 0.0f32;
        for j in 0..KV_TILE_SZ {
            scores[j] = (scores[j] - max_score).exp();
            sum_exp += scores[j];
        }

        // Normalize
        for j in 0..KV_TILE_SZ {
            scores[j] /= sum_exp;
        }

        time_softmax += start.elapsed();

        // Reset scores for next iteration
        for j in 0..KV_TILE_SZ {
            scores[j] = (j as f32 * 0.1).sin();
        }
    }
    println!("  Time: {:?}", time_softmax / N_ITERATIONS as u32);
    println!("  Ops: {:.1}M ops/s",
             (KV_TILE_SZ * 3) as f64 * N_ITERATIONS as f64 / time_softmax.as_secs_f64() / 1e6);

    // Test 4: Softmax with NEON
    println!("\n4. Softmax with NEON (if available):");
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        let mut time_softmax_neon = Duration::new(0, 0);
        for _ in 0..N_ITERATIONS {
            let start = Instant::now();

            unsafe {
                // Find max with NEON
                let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
                for j in (0..KV_TILE_SZ).step_by(4) {
                    let s = vld1q_f32(scores[j..].as_ptr());
                    max_vec = vmaxq_f32(max_vec, s);
                }
                let max_arr: [f32; 4] = std::mem::transmute(max_vec);
                let max_score = max_arr[0].max(max_arr[1]).max(max_arr[2]).max(max_arr[3]);

                // Exp and sum with NEON
                let max_vec = vdupq_n_f32(max_score);
                let mut sum_vec = vdupq_n_f32(0.0);
                for j in (0..KV_TILE_SZ).step_by(4) {
                    let s = vld1q_f32(scores[j..].as_ptr());
                    let diff = vsubq_f32(s, max_vec);
                    let diff_arr: [f32; 4] = std::mem::transmute(diff);
                    for jj in 0..4 {
                        scores[j + jj] = diff_arr[jj].exp();
                    }
                    let exp_vec = vld1q_f32(scores[j..].as_ptr());
                    sum_vec = vaddq_f32(sum_vec, exp_vec);
                }
                let sum_arr: [f32; 4] = std::mem::transmute(sum_vec);
                let sum_exp = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

                // Normalize with NEON
                let sum_vec = vdupq_n_f32(1.0 / sum_exp);
                for j in (0..KV_TILE_SZ).step_by(4) {
                    let s = vld1q_f32(scores[j..].as_ptr());
                    let normalized = vmulq_f32(s, sum_vec);
                    vst1q_f32(scores[j..].as_mut_ptr(), normalized);
                }
            }

            time_softmax_neon += start.elapsed();

            // Reset
            for j in 0..KV_TILE_SZ {
                scores[j] = (j as f32 * 0.1).sin();
            }
        }
        println!("  Time: {:?}", time_softmax_neon / N_ITERATIONS as u32);
        println!("  Speedup vs scalar: {:.1}x",
                 time_softmax.as_secs_f64() / time_softmax_neon.as_secs_f64());
    }

    #[cfg(not(target_arch = "aarch64"))]
    println!("  Not available (not aarch64)");

    println!("\n=== Analysis ===");
    println!("The Load K/V tile involves transposition (stride-64 access)");
    println!("This is expensive and not easily optimized with SIMD");
    println!("Softmax involves exp() which is also expensive");
}
