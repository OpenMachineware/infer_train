// Benchmark IQ3_XXS v1 vs v2 performance
// Tests whether inline assembly optimizations improve performance

use infer_train::quant::types::{BlockIQ3XXS, BlockQ8K, QK_K};
use infer_train::quant::vec_dot::arm::{
    vec_dot_iq3_xxs_q8_k_neon, vec_dot_iq3_xxs_q8_k_neon_v2,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;

fn main() {
    let n_blocks = 1024; // More blocks for better accuracy
    let n = n_blocks * QK_K;
    let iterations = 100; // Fewer iterations, more data per iteration

    // Create test data
    let mut rng = StdRng::seed_from_u64(42);
    let mut x_blocks = Vec::with_capacity(n_blocks);
    let mut y_blocks = Vec::with_capacity(n_blocks);

    for _ in 0..n_blocks {
        let mut qs = [0u8; 96]; // 3*QK_K/8 = 96
        for byte in qs.iter_mut() {
            *byte = rng.r#gen::<u8>();
        }

        let d_f32 = rng.r#gen::<f32>();
        let d_bits = d_f32.to_bits(); // Convert to u32
        let d = (d_bits >> 16) as u16; // Take upper 16 bits as FP16 approximation

        x_blocks.push(BlockIQ3XXS { d, qs });
    }

    for _ in 0..n_blocks {
        let mut qs = [0i8; QK_K];
        rng.fill(&mut qs);
        let d = rng.r#gen::<f32>();
        let bsums = [0i16; 16];

        y_blocks.push(BlockQ8K { d, qs, bsums });
    }

    let x: &[BlockIQ3XXS] = &x_blocks;
    let y: &[BlockQ8K] = &y_blocks;

    // Warm up
    for _ in 0..100 {
        unsafe {
            let _ = vec_dot_iq3_xxs_q8_k_neon(n, x, y);
            let _ = vec_dot_iq3_xxs_q8_k_neon_v2(n, x, y);
        }
    }

    // Benchmark v1
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            let result = vec_dot_iq3_xxs_q8_k_neon(n, x, y);
            std::hint::black_box(result);
        }
    }
    let elapsed_v1 = start.elapsed();

    // Benchmark v2
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            let result = vec_dot_iq3_xxs_q8_k_neon_v2(n, x, y);
            std::hint::black_box(result);
        }
    }
    let elapsed_v2 = start.elapsed();

    let ns_v1 = elapsed_v1.as_nanos() as f64 / iterations as f64;
    let ns_v2 = elapsed_v2.as_nanos() as f64 / iterations as f64;
    let ratio = ns_v1 / ns_v2 * 100.0;

    println!("IQ3_XXS Performance Comparison");
    println!("================================");
    println!("Blocks: {}, Iterations: {}", n_blocks, iterations);
    println!();
    println!("Version   | Time (ns) | Ratio");
    println!("----------|-----------|-------");
    println!("v1        | {:9.1} | 100.0%", ns_v1);
    println!("v2 (opt)  | {:9.1} | {:.1}%", ns_v2, ratio);
    println!();
    if ns_v2 < ns_v1 {
        let improvement = (ns_v1 - ns_v2) / ns_v1 * 100.0;
        println!("✓ v2 is {:.1}% faster than v1", improvement);
    } else {
        let regression = (ns_v2 - ns_v1) / ns_v1 * 100.0;
        println!("✗ v2 is {:.1}% slower than v1 (regression)", regression);
    }
}
