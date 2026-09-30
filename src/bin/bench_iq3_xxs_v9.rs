// Benchmark IQ3_XXS v9 (exact llama.cpp NEON intrinsics copy) vs llama.cpp

use infer_train::quant::types::{BlockIQ3XXS, BlockQ8K, QK_K};
use infer_train::quant::vec_dot::arm::vec_dot_iq3_xxs_q8_k_neon_v9;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;

unsafe extern "C" {
    fn llama_vec_dot_iq3_xxs_q8_K(
        n: i32,
        s: *mut f32,
        x: *const BlockIQ3XXS,
        y: *const BlockQ8K,
    );
}

fn main() {
    let n_blocks = 256;
    let n = n_blocks * QK_K;
    let iterations = 10;
    let warmup = 100;

    // Create test data
    let mut rng = StdRng::seed_from_u64(42);
    let mut x_blocks = Vec::with_capacity(n_blocks);
    let mut y_blocks = Vec::with_capacity(n_blocks);

    for _ in 0..n_blocks {
        let mut qs = [0u8; 96];
        for byte in qs.iter_mut() {
            *byte = rng.r#gen::<u8>();
        }
        let d_f32 = 1.0f32 + rng.r#gen::<f32>() * 2.0;
        let d = d_f32.to_bits() as u16;

        x_blocks.push(BlockIQ3XXS { d, qs });
    }

    for _ in 0..n_blocks {
        let mut qs = [0i8; QK_K];
        for byte in qs.iter_mut() {
            *byte = (rng.r#gen::<u8>() as i8) - 64;
        }
        let d = 1.0f32 + rng.r#gen::<f32>() * 2.0;
        let bsums = [0i16; 16];

        y_blocks.push(BlockQ8K { d, qs, bsums });
    }

    let x: &[BlockIQ3XXS] = &x_blocks;
    let y: &[BlockQ8K] = &y_blocks;

    println!("=== IQ3_XXS v9 (exact llama.cpp NEON intrinsics copy) ===");
    println!(
        "Method: {} iterations, {} warmup, same process",
        iterations, warmup
    );
    println!();

    // Warmup for llama.cpp
    for _ in 0..warmup {
        unsafe {
            let mut result = 0.0f32;
            llama_vec_dot_iq3_xxs_q8_K(
                n as i32,
                &mut result,
                x.as_ptr(),
                y.as_ptr(),
            );
            std::hint::black_box(result);
        }
    }

    // Benchmark llama.cpp
    let mut llama_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        unsafe {
            let mut result = 0.0f32;
            llama_vec_dot_iq3_xxs_q8_K(
                n as i32,
                &mut result,
                x.as_ptr(),
                y.as_ptr(),
            );
            std::hint::black_box(result);
        }
        llama_times.push(start.elapsed().as_nanos() as f64);
    }

    // Warmup for Rust v9
    for _ in 0..warmup {
        unsafe {
            let result = vec_dot_iq3_xxs_q8_k_neon_v9(n, x, y);
            std::hint::black_box(result);
        }
    }

    // Benchmark Rust v9
    let mut rust_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        unsafe {
            let result = vec_dot_iq3_xxs_q8_k_neon_v9(n, x, y);
            std::hint::black_box(result);
        }
        rust_times.push(start.elapsed().as_nanos() as f64);
    }

    // Calculate statistics
    llama_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rust_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let llama_median = llama_times[llama_times.len() / 2];
    let rust_median = rust_times[rust_times.len() / 2];

    println!("llama.cpp median: {:.0} ns", llama_median);
    println!("Rust v9 median:   {:.0} ns", rust_median);
    println!("Ratio:            {:.1}%", rust_median / llama_median * 100.0);
}
