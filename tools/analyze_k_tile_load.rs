#!/usr/bin/env cargo
//! Measure actual K tile load time and compare approaches

use std::time::{Duration, Instant};

fn main() {
    const HEAD_DIM: usize = 128;
    const KV_TILE_SZ: usize = 64;
    const N_TOKENS: usize = 64;
    const N_KV_HEADS: usize = 2;
    const N_ITERATIONS: usize = 1000;

    println!("=== K Tile Load Performance Analysis ===\n");

    // Simulate K tensor: [n_tokens, n_kv_heads, head_dim]
    let kv_size = N_KV_HEADS * HEAD_DIM;
    let k_tensor: Vec<f32> = (0..N_TOKENS * kv_size).map(|i| i as f32 * 0.001).collect();
    let mut k_tile: Vec<f32> = vec![0.0; HEAD_DIM * KV_TILE_SZ];

    // Test 1: Current approach (transpose during load)
    println!("1. Current approach - transpose during load:");
    let mut time_transpose = Duration::new(0, 0);
    for _ in 0..N_ITERATIONS {
        let kv_head = 0;
        let kv_tile_start = 0;

        let start = Instant::now();
        for tk in 0..KV_TILE_SZ {
            let kv_pos = kv_tile_start + tk;
            let k_offset = kv_pos * kv_size + kv_head * HEAD_DIM;

            for d in 0..HEAD_DIM {
                k_tile[d * KV_TILE_SZ + tk] = k_tensor[k_offset + d];
            }
        }
        time_transpose += start.elapsed();
    }

    let bytes_loaded = HEAD_DIM * KV_TILE_SZ * 4 * N_ITERATIONS;
    println!("  Time: {:?}", time_transpose / N_ITERATIONS as u32);
    println!("  Bandwidth: {:.1} GB/s", bytes_loaded as f64 / time_transpose.as_secs_f64() / 1e9);

    // Test 2: Non-transposed load (direct copy)
    println!("\n2. Non-transposed load - direct copy:");
    let mut k_tile_notranspose: Vec<f32> = vec![0.0; KV_TILE_SZ * HEAD_DIM];
    let mut time_direct = Duration::new(0, 0);
    for _ in 0..N_ITERATIONS {
        let kv_head = 0;
        let kv_tile_start = 0;

        let start = Instant::now();
        for tk in 0..KV_TILE_SZ {
            let kv_pos = kv_tile_start + tk;
            let k_offset = kv_pos * kv_size + kv_head * HEAD_DIM;

            // Direct copy, no transpose
            k_tile_notranspose[tk * HEAD_DIM..(tk + 1) * HEAD_DIM]
                .copy_from_slice(&k_tensor[k_offset..k_offset + HEAD_DIM]);
        }
        time_direct += start.elapsed();
    }

    println!("  Time: {:?}", time_direct / N_ITERATIONS as u32);
    println!("  Bandwidth: {:.1} GB/s", bytes_loaded as f64 / time_direct.as_secs_f64() / 1e9);
    println!("  Speedup vs transpose: {:.1}x", time_transpose.as_secs_f64() / time_direct.as_secs_f64());

    // Test 3: Transpose using SIMD
    println!("\n3. Transpose with NEON (4x4 blocks):");
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        let mut time_neon = Duration::new(0, 0);
        for _ in 0..N_ITERATIONS {
            let kv_head = 0;
            let kv_tile_start = 0;

            let start = Instant::now();
            unsafe {
                for tk in (0..KV_TILE_SZ).step_by(4) {
                    for d in (0..HEAD_DIM).step_by(4) {
                        // Load 4x4 block
                        let mut rows: [float32x4_t; 4] = std::mem::zeroed();
                        for ii in 0..4 {
                            let tk_pos = kv_tile_start + tk + ii;
                            let k_offset = tk_pos * kv_size + kv_head * HEAD_DIM + d;
                            rows[ii] = vld1q_f32(k_tensor[k_offset..].as_ptr());
                        }

                        // Transpose 4x4
                        let t0 = vzip1q_f32(rows[0], rows[2]);
                        let t1 = vzip2q_f32(rows[0], rows[2]);
                        let t2 = vzip1q_f32(rows[1], rows[3]);
                        let t3 = vzip2q_f32(rows[1], rows[3]);

                        let r0 = vzip1q_f32(t0, t2);
                        let r1 = vzip2q_f32(t0, t2);
                        let r2 = vzip1q_f32(t1, t3);
                        let r3 = vzip2q_f32(t1, t3);

                        // Store transposed
                        vst1q_f32(k_tile.as_mut_ptr().add((d + 0) * KV_TILE_SZ + tk), r0);
                        vst1q_f32(k_tile.as_mut_ptr().add((d + 1) * KV_TILE_SZ + tk), r1);
                        vst1q_f32(k_tile.as_mut_ptr().add((d + 2) * KV_TILE_SZ + tk), r2);
                        vst1q_f32(k_tile.as_mut_ptr().add((d + 3) * KV_TILE_SZ + tk), r3);
                    }
                }
            }
            time_neon += start.elapsed();
        }

        println!("  Time: {:?}", time_neon / N_ITERATIONS as u32);
        println!("  Bandwidth: {:.1} GB/s", bytes_loaded as f64 / time_neon.as_secs_f64() / 1e9);
        println!("  Speedup vs scalar transpose: {:.1}x", time_transpose.as_secs_f64() / time_neon.as_secs_f64());
    }

    #[cfg(not(target_arch = "aarch64"))]
    println!("  Not available (not aarch64)");

    println!("\n=== Analysis ===");
    println!("Transposing during load causes stride-64 writes");
    println!("This is slow because each write evicts cache line");
    println!("llama.cpp also transposes but may have cache advantages");
}
