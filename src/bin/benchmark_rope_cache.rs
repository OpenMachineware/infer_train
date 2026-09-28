// Benchmark: RoPE with cache vs without cache
// Run: cargo run --release --bin benchmark_rope_cache

use infer_train::ops::rope::{
    RopeCache, RopeParams, rope_neox, rope_neox_batch_with_cache,
    rope_neox_with_cache,
};
use std::time::Instant;

fn main() {
    println!("=== RoPE: Precomputed Cache vs No Cache ===\n");

    let n_dims = 128;
    let params = RopeParams { n_dims, ..Default::default() };

    // Test different batch sizes
    let batch_sizes = [64, 128, 256, 512, 1024];

    println!("Single position (batch=1):");
    println!(
        "{:8} | {:15} | {:15} | {:8}",
        "n_dims", "Cache (ns)", "No cache (ns)", "Speedup"
    );
    println!("{}", "-".repeat(55));

    let src: Vec<f32> = (0..n_dims).map(|i| (i as f32 + 1.0) / 100.0).collect();
    let mut dst = vec![0.0f32; n_dims];
    let mut cache = RopeCache::new(n_dims);

    let iterations = 10000;

    // Warmup
    for pos in 0..10 {
        cache.init(pos, &params);
        rope_neox_with_cache(&cache, &src, &mut dst);
        rope_neox(&src, &mut dst, pos, &params);
    }

    // Benchmark with cache
    let start = Instant::now();
    for pos in 0..iterations {
        cache.init(pos % 100, &params);
        rope_neox_with_cache(&cache, &src, &mut dst);
    }
    let t_cache = start.elapsed().as_secs_f64() / iterations as f64 * 1e9;

    // Benchmark without cache
    let start = Instant::now();
    for pos in 0..iterations {
        rope_neox(&src, &mut dst, pos % 100, &params);
    }
    let t_no_cache = start.elapsed().as_secs_f64() / iterations as f64 * 1e9;

    let speedup = t_no_cache / t_cache;
    println!(
        "{:8} | {:15.2} | {:15.2} | {:8.2}x",
        n_dims, t_cache, t_no_cache, speedup
    );

    // Batch processing
    println!("\nBatch processing:");
    println!(
        "{:8} | {:15} | {:15} | {:8}",
        "Batch", "Cache (us)", "No cache (us)", "Speedup"
    );
    println!("{}", "-".repeat(55));

    for &batch in &batch_sizes {
        let positions: Vec<i32> = (0..batch as i32).collect();
        let src: Vec<f32> =
            (0..batch * n_dims).map(|i| (i as f32 + 1.0) / 100.0).collect();
        let mut dst = vec![0.0f32; batch * n_dims];

        // Warmup
        rope_neox_batch_with_cache(&src, &mut dst, &positions, &params);

        // Benchmark with cache
        let iterations = (10000 / batch).max(100);
        let start = Instant::now();
        for _ in 0..iterations {
            rope_neox_batch_with_cache(&src, &mut dst, &positions, &params);
        }
        let t_cache = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;

        // Benchmark without cache (using rope_neox directly)
        let start = Instant::now();
        for _ in 0..iterations {
            for (i, &pos) in positions.iter().enumerate() {
                rope_neox(
                    &src[i * n_dims..],
                    &mut dst[i * n_dims..],
                    pos as usize,
                    &params,
                );
            }
        }
        let t_no_cache =
            start.elapsed().as_secs_f64() / iterations as f64 * 1e6;

        let speedup = t_no_cache / t_cache;
        println!(
            "{:8} | {:15.2} | {:15.2} | {:8.2}x",
            batch, t_cache, t_no_cache, speedup
        );
    }

    // Correctness
    println!("\n=== Correctness ===");
    let test_src: Vec<f32> =
        (0..n_dims).map(|i| (i as f32 + 1.0) / 100.0).collect();
    let mut dst_cache = vec![0.0f32; n_dims];
    let mut dst_no_cache = vec![0.0f32; n_dims];

    let mut cache = RopeCache::new(n_dims);

    for pos in 0..10 {
        cache.init(pos, &params);
        rope_neox_with_cache(&cache, &test_src, &mut dst_cache);
        rope_neox(&test_src, &mut dst_no_cache, pos, &params);

        let max_diff = (0..n_dims)
            .map(|i| (dst_cache[i] - dst_no_cache[i]).abs())
            .fold(0.0f32, f32::max);
        println!("Position {}: max_diff = {:.2e}", pos, max_diff);
    }
}
