// Benchmark: GELU lookup table vs exp approximation
// Run: cargo run --release --bin benchmark_gelu_table

use infer_train::ops::{gelu_f32_table, gelu_quick_f32_table};
use infer_train::ops::gelu::{gelu_f32, gelu_quick_f32};
use std::time::Instant;

fn main() {
    println!("=== GELU: FP16 Lookup Table vs Exp Approximation ===\n");

    let sizes = [256, 512, 1024, 2048, 4096, 8192];
    let iterations = 10000;

    println!("GELU Performance:");
    println!("{:8} | {:15} | {:15} | {:8}", "Size", "Table (us)", "Exp (us)", "Speedup");
    println!("{}", "-".repeat(55));

    for &size in &sizes {
        let x: Vec<f32> = (0..size).map(|i| (i as f32 - size as f32 / 2.0) * 0.01).collect();
        let mut dst_table = vec![0.0f32; size];
        let mut dst_exp = vec![0.0f32; size];

        // Warmup
        for _ in 0..100 {
            gelu_f32_table(&x, &mut dst_table);
            unsafe { gelu_f32(&x, &mut dst_exp); }
        }

        // Benchmark table
        let start = Instant::now();
        for _ in 0..iterations {
            gelu_f32_table(&x, &mut dst_table);
        }
        let t_table = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;

        // Benchmark exp
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe { gelu_f32(&x, &mut dst_exp); }
        }
        let t_exp = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;

        let speedup = t_exp / t_table;
        println!("{:8} | {:15.2} | {:15.2} | {:8.2}x", size, t_table, t_exp, speedup);
    }

    println!("\nGELU-Quick Performance:");
    println!("{:8} | {:15} | {:15} | {:8}", "Size", "Table (us)", "Exp (us)", "Speedup");
    println!("{}", "-".repeat(55));

    for &size in &sizes {
        let x: Vec<f32> = (0..size).map(|i| (i as f32 - size as f32 / 2.0) * 0.01).collect();
        let mut dst_table = vec![0.0f32; size];
        let mut dst_exp = vec![0.0f32; size];

        // Warmup
        for _ in 0..100 {
            gelu_quick_f32_table(&x, &mut dst_table);
            unsafe { gelu_quick_f32(&x, &mut dst_exp); }
        }

        // Benchmark table
        let start = Instant::now();
        for _ in 0..iterations {
            gelu_quick_f32_table(&x, &mut dst_table);
        }
        let t_table = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;

        // Benchmark exp
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe { gelu_quick_f32(&x, &mut dst_exp); }
        }
        let t_exp = start.elapsed().as_secs_f64() / iterations as f64 * 1e6;

        let speedup = t_exp / t_table;
        println!("{:8} | {:15.2} | {:15.2} | {:8.2}x", size, t_table, t_exp, speedup);
    }

    // Correctness
    println!("\n=== Correctness ===");
    let test_vals: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let mut dst_table = vec![0.0f32; test_vals.len()];
    let mut dst_exp = vec![0.0f32; test_vals.len()];

    gelu_f32_table(&test_vals, &mut dst_table);
    unsafe { gelu_f32(&test_vals, &mut dst_exp); }

    let mut max_diff = 0.0f32;
    for i in 0..test_vals.len() {
        let diff = (dst_table[i] - dst_exp[i]).abs();
        max_diff = max_diff.max(diff);
    }
    println!("GELU max diff (table vs exp): {}", max_diff);

    gelu_quick_f32_table(&test_vals, &mut dst_table);
    unsafe { gelu_quick_f32(&test_vals, &mut dst_exp); }

    max_diff = 0.0f32;
    for i in 0..test_vals.len() {
        let diff = (dst_table[i] - dst_exp[i]).abs();
        max_diff = max_diff.max(diff);
    }
    println!("GELU-Quick max diff (table vs exp): {}", max_diff);
}
