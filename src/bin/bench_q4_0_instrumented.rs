// Instrumented benchmark for Q4_0: measure each step
// Use Instant for precise timing

use std::hint::black_box;
use std::time::Instant;

use infer_train::quant::types::*;
use infer_train::quant::vec_dot::arm::*;

unsafe extern "C" {
    fn llama_vec_dot_q4_0_q8_0(
        n: i32,
        s: *mut f32,
        x: *const BlockQ4_0,
        y: *const BlockQ8_0,
    );
}

fn main() {
    let n_blocks = 1024; // 4x more data for stability
    let n = n_blocks * QK8_0;
    let iterations = 1000; // 10x fewer iterations to balance time
    let warmup = 100;

    // Create test data
    let mut x_blocks: Vec<BlockQ4_0> = Vec::with_capacity(n_blocks);
    let mut y_blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_blocks);

    for i in 0..n_blocks {
        let mut qs = [0u8; QK8_0 / 2];
        let mut q8 = [0i8; QK8_0];
        for j in 0..QK8_0 / 2 {
            qs[j] = ((i * 17 + j * 7) % 256) as u8;
        }
        for j in 0..QK8_0 {
            q8[j] = ((i * 13 + j * 11) % 256 - 128) as i8;
        }
        let d = half::f16::from_f32(1.0 + (i % 10) as f32).to_bits();

        x_blocks.push(BlockQ4_0 { d, qs });
        y_blocks.push(BlockQ8_0 { d, qs: q8 });
    }

    let x: &[BlockQ4_0] = &x_blocks;
    let y: &[BlockQ8_0] = &y_blocks;

    // Warm up
    for _ in 0..warmup {
        let mut result = 0.0f32;
        unsafe {
            llama_vec_dot_q4_0_q8_0(
                n as i32,
                &mut result,
                x.as_ptr(),
                y.as_ptr(),
            );
        }
        let _ = black_box(unsafe { vec_dot_q4_0_q8_0_neon(n, x, y) });
    }

    println!("=== Q4_0 Instrumented Benchmark ===");
    println!("Blocks: {}, Iterations: {}", n_blocks, iterations);
    println!();

    // Measure llama.cpp total time
    let start = Instant::now();
    for _ in 0..iterations {
        let mut result = 0.0f32;
        unsafe {
            llama_vec_dot_q4_0_q8_0(
                n as i32,
                &mut result,
                x.as_ptr(),
                y.as_ptr(),
            );
            black_box(result);
        }
    }
    let llama_time = start.elapsed();

    // Measure Rust total time
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            let result = vec_dot_q4_0_q8_0_neon(n, x, y);
            black_box(result);
        }
    }
    let rust_time = start.elapsed();

    println!("Total time:");
    println!(
        "  llama.cpp: {:.2} ns/iter",
        llama_time.as_nanos() as f64 / iterations as f64
    );
    println!(
        "  Rust:      {:.2} ns/iter",
        rust_time.as_nanos() as f64 / iterations as f64
    );
    println!(
        "  Ratio:     {:.1}%",
        rust_time.as_nanos() as f64 / llama_time.as_nanos() as f64 * 100.0
    );
}
