#!/usr/bin/env cargo
//! Detailed analysis of K tile loading with cache behavior
//! Test multiple iterations and check for cache effects

use std::arch::aarch64::*;
use std::time::{Duration, Instant};

const KV_TILE_SZ: usize = 64;
const HEAD_DIM: usize = 128;

struct ScratchSpace {
    k_tile: Vec<f32>,
}

impl ScratchSpace {
    fn new() -> Self {
        Self {
            k_tile: vec![0.0; HEAD_DIM * KV_TILE_SZ],
        }
    }
}

/// Standard approach (both ours and llama.cpp use this)
fn load_k_tile_standard(
    k_outputs: &[f32],
    kv_size_per_token: usize,
    kv_head: usize,
    kv_tile_start: usize,
    kv_tile_size: usize,
    scratch: &mut ScratchSpace,
) {
    for tk in 0..kv_tile_size {
        let kv_pos = kv_tile_start + tk;
        let k_offset = kv_pos * kv_size_per_token + kv_head * HEAD_DIM;

        for d in 0..HEAD_DIM {
            scratch.k_tile[d * KV_TILE_SZ + tk] = k_outputs[k_offset + d];
        }
    }
}

/// Swapped loop order
fn load_k_tile_swapped(
    k_outputs: &[f32],
    kv_size_per_token: usize,
    kv_head: usize,
    kv_tile_start: usize,
    kv_tile_size: usize,
    scratch: &mut ScratchSpace,
) {
    for d in 0..HEAD_DIM {
        for tk in 0..kv_tile_size {
            let kv_pos = kv_tile_start + tk;
            let k_offset = kv_pos * kv_size_per_token + kv_head * HEAD_DIM + d;
            scratch.k_tile[d * KV_TILE_SZ + tk] = k_outputs[k_offset];
        }
    }
}

/// NEON SIMD version
fn load_k_tile_simd(
    k_outputs: &[f32],
    kv_size_per_token: usize,
    kv_head: usize,
    kv_tile_start: usize,
    kv_tile_size: usize,
    scratch: &mut ScratchSpace,
) {
    unsafe {
        for tk in 0..kv_tile_size {
            let kv_pos = kv_tile_start + tk;
            let k_offset = kv_pos * kv_size_per_token + kv_head * HEAD_DIM;

            for d in (0..HEAD_DIM).step_by(4) {
                let k_vec = vld1q_f32(k_outputs.as_ptr().add(k_offset + d));
                let k_arr: [f32; 4] = std::mem::transmute(k_vec);
                let dst_base = scratch.k_tile.as_mut_ptr().add(d * KV_TILE_SZ + tk);
                for i in 0..4 {
                    *dst_base.add(i * KV_TILE_SZ) = k_arr[i];
                }
            }
        }
    }
}

/// Measure time with checksum to prevent dead-code elimination
fn measure_time<F>(f: F, iterations: usize) -> (Duration, f32)
where
    F: Fn(&mut ScratchSpace),
{
    let mut scratch = ScratchSpace::new();
    let mut checksum: f32 = 0.0;

    let start = Instant::now();
    for _ in 0..iterations {
        f(&mut scratch);
        // Use black_box to prevent compiler from optimizing away the store
        std::hint::black_box(&mut scratch.k_tile);
        checksum += scratch.k_tile[0];
    }
    let duration = start.elapsed();

    // Use black_box on checksum too
    std::hint::black_box(checksum);

    (duration, checksum)
}

fn main() {
    let n_tokens = 64;
    let n_kv_heads = 8;
    let kv_size_per_token = n_kv_heads * HEAD_DIM;
    let total_kv_size = kv_size_per_token * n_tokens;

    let k_outputs: Vec<f32> = (0..total_kv_size)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    println!("=== K Tile Loading Cache Analysis ===");
    println!("Tile size: {} x {} = {} elements ({} KB)\n",
             HEAD_DIM, KV_TILE_SZ, HEAD_DIM * KV_TILE_SZ, HEAD_DIM * KV_TILE_SZ * 4 / 1024);

    let kv_head = 0;
    let kv_tile_start = 0;
    let kv_tile_size = KV_TILE_SZ;
    let iterations = 10000;

    // Test each approach multiple times
    println!("Testing standard approach 10 times:");
    let mut times_standard = Vec::new();
    for i in 0..10 {
        let (time, _) = measure_time(|s| {
            load_k_tile_standard(&k_outputs, kv_size_per_token, kv_head, kv_tile_start, kv_tile_size, s);
        }, iterations);
        println!("  Run {}: {:?}", i+1, time);
        times_standard.push(time);
    }

    println!("\nTesting swapped approach 10 times:");
    let mut times_swapped = Vec::new();
    for i in 0..10 {
        let (time, _) = measure_time(|s| {
            load_k_tile_swapped(&k_outputs, kv_size_per_token, kv_head, kv_tile_start, kv_tile_size, s);
        }, iterations);
        println!("  Run {}: {:?}", i+1, time);
        times_swapped.push(time);
    }

    println!("\nTesting SIMD approach 10 times:");
    let mut times_simd = Vec::new();
    for i in 0..10 {
        let (time, _) = measure_time(|s| {
            load_k_tile_simd(&k_outputs, kv_size_per_token, kv_head, kv_tile_start, kv_tile_size, s);
        }, iterations);
        println!("  Run {}: {:?}", i+1, time);
        times_simd.push(time);
    }

    // Calculate averages
    let avg_standard: Duration = times_standard.iter().sum::<Duration>() / 10;
    let avg_swapped: Duration = times_swapped.iter().sum::<Duration>() / 10;
    let avg_simd: Duration = times_simd.iter().sum::<Duration>() / 10;

    println!("\n=== Average Times ===");
    println!("Standard: {:?}", avg_standard);
    println!("Swapped:  {:?}", avg_swapped);
    println!("SIMD:     {:?}", avg_simd);

    // Calculate throughput
    let bytes_per_iter = HEAD_DIM * KV_TILE_SZ * 4;
    let gb_standard = bytes_per_iter as f64 * iterations as f64 / avg_standard.as_secs_f64() / 1e9;
    let gb_swapped = bytes_per_iter as f64 * iterations as f64 / avg_swapped.as_secs_f64() / 1e9;
    let gb_simd = bytes_per_iter as f64 * iterations as f64 / avg_simd.as_secs_f64() / 1e9;

    println!("\n=== Throughput ===");
    println!("Standard: {:.2} GB/s", gb_standard);
    println!("Swapped:  {:.2} GB/s", gb_swapped);
    println!("SIMD:     {:.2} GB/s", gb_simd);

    println!("\n=== Speedup vs Standard ===");
    println!("Swapped: {:.2}x", avg_standard.as_secs_f64() / avg_swapped.as_secs_f64());
    println!("SIMD:    {:.2}x", avg_standard.as_secs_f64() / avg_simd.as_secs_f64());
}
