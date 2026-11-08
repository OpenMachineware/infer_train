#!/usr/bin/env cargo
//! Compare isolated K tile loading vs actual code in flash_attention.rs
//! Find why real code is 1.9x slower than benchmark

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

/// Isolated benchmark (like previous test)
fn load_k_tile_isolated(
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

/// Actual code from flash_attention.rs (simulated with slices)
fn load_k_tile_with_slices(
    k_outputs: &[f32],
    kv_size: usize,  // Note: this is passed as parameter, not computed locally
    kv_head: usize,
    kv_tile_start: usize,
    kv_tile_size: usize,
    scratch: &mut ScratchSpace,
) {
    // Create slice inside the function (like real code does)
    let k_slice = k_outputs;
    let k_tile_slice = &mut scratch.k_tile;

    for tk in 0..kv_tile_size {
        let kv_pos = kv_tile_start + tk;
        // Note: Real code computes k_offset like this
        let k_offset = kv_pos * kv_size + kv_head * HEAD_DIM;

        for d in 0..HEAD_DIM {
            // Real code accesses slices with bounds checking
            k_tile_slice[d * KV_TILE_SZ + tk] = k_slice[k_offset + d];
        }
    }
}

/// With unchecked array access (like what optimized code should do)
fn load_k_tile_unchecked(
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

            for d in 0..HEAD_DIM {
                *scratch.k_tile.as_mut_ptr().add(d * KV_TILE_SZ + tk) =
                    *k_outputs.as_ptr().add(k_offset + d);
            }
        }
    }
}

/// Measure time
fn measure_time<F>(f: F, iterations: usize) -> Duration
where
    F: Fn(&mut ScratchSpace),
{
    let mut scratch = ScratchSpace::new();

    // Warmup
    for _ in 0..100 {
        f(&mut scratch);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f(&mut scratch);
        std::hint::black_box(&scratch.k_tile);
    }
    start.elapsed()
}

fn main() {
    let n_tokens = 64;
    let n_kv_heads = 8;
    let kv_size_per_token = n_kv_heads * HEAD_DIM;
    let total_kv_size = kv_size_per_token * n_tokens;

    let k_outputs: Vec<f32> = (0..total_kv_size)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();

    println!("=== K Tile Loading: Isolated vs Real Code Simulation ===");
    println!("Tile size: {} x {} = {} elements ({} KB)\n",
             HEAD_DIM, KV_TILE_SZ, HEAD_DIM * KV_TILE_SZ, HEAD_DIM * KV_TILE_SZ * 4 / 1024);

    let kv_head = 0;
    let kv_tile_start = 0;
    let kv_tile_size = KV_TILE_SZ;
    let iterations = 10000;

    // Test isolated approach
    println!("Testing isolated approach:");
    let time_isolated = measure_time(|s| {
        load_k_tile_isolated(&k_outputs, kv_size_per_token, kv_head, kv_tile_start, kv_tile_size, s);
    }, iterations);
    println!("  Time: {:?}", time_isolated);

    // Test with slices (like real code)
    println!("\nTesting with slices (like real code):");
    let time_slices = measure_time(|s| {
        load_k_tile_with_slices(&k_outputs, kv_size_per_token, kv_head, kv_tile_start, kv_tile_size, s);
    }, iterations);
    println!("  Time: {:?}", time_slices);

    // Test unchecked version
    println!("\nTesting unchecked (pointer-based):");
    let time_unchecked = measure_time(|s| {
        load_k_tile_unchecked(&k_outputs, kv_size_per_token, kv_head, kv_tile_start, kv_tile_size, s);
    }, iterations);
    println!("  Time: {:?}", time_unchecked);

    // Calculate throughput
    let bytes_per_iter = HEAD_DIM * KV_TILE_SZ * 4;
    let gb_isolated = bytes_per_iter as f64 * iterations as f64 / time_isolated.as_secs_f64() / 1e9;
    let gb_slices = bytes_per_iter as f64 * iterations as f64 / time_slices.as_secs_f64() / 1e9;
    let gb_unchecked = bytes_per_iter as f64 * iterations as f64 / time_unchecked.as_secs_f64() / 1e9;

    println!("\n=== Throughput ===");
    println!("Isolated:   {:.2} GB/s", gb_isolated);
    println!("Slices:     {:.2} GB/s", gb_slices);
    println!("Unchecked:  {:.2} GB/s", gb_unchecked);

    println!("\n=== Speedup vs Isolated ===");
    println!("Slices:     {:.2}x", time_isolated.as_secs_f64() / time_slices.as_secs_f64());
    println!("Unchecked:  {:.2}x", time_isolated.as_secs_f64() / time_unchecked.as_secs_f64());
}
