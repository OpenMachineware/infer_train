// Benchmark SiLU and GELU vs llama.cpp
use std::arch::aarch64::*;
use std::time::Instant;

const GELU_COEF_A: f32 = 0.044715;
const SQRT_2_OVER_PI: f32 = 0.7978845608028654;

// Our NEON SiLU
fn silu_neon(x: &[f32], dst: &mut [f32]) {
    unsafe {
        infer_train::ops::silu::silu_f32(x, dst);
    }
}

// llama.cpp-style scalar SiLU
fn silu_scalar(x: &[f32], dst: &mut [f32]) {
    for i in 0..x.len() {
        dst[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

// llama.cpp-style vectorized SiLU (using our vexp)
fn silu_llama_cpp(x: &[f32], dst: &mut [f32]) {
    unsafe {
        let n = x.len();
        let chunks = n / 4;
        let one = vdupq_n_f32(1.0);
        let zero = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let vx = vld1q_f32(x.as_ptr().add(i * 4));
            let neg_x = vsubq_f32(zero, vx);
            let exp_neg_x = infer_train::ops::silu::vexpq_f32(neg_x);
            let one_plus_exp = vaddq_f32(one, exp_neg_x);
            let result = vdivq_f32(vx, one_plus_exp);
            vst1q_f32(dst.as_mut_ptr().add(i * 4), result);
        }

        for i in (chunks * 4)..n {
            dst[i] = x[i] / (1.0 + (-x[i]).exp());
        }
    }
}

// Our NEON GELU
fn gelu_neon(x: &[f32], dst: &mut [f32]) {
    unsafe {
        infer_train::ops::gelu::gelu_f32(x, dst);
    }
}

// llama.cpp-style scalar GELU
fn gelu_scalar(x: &[f32], dst: &mut [f32]) {
    for i in 0..x.len() {
        let inner = 1.0 + GELU_COEF_A * x[i] * x[i];
        let tanh_arg = SQRT_2_OVER_PI * x[i] * inner;
        dst[i] = 0.5 * x[i] * (1.0 + tanh_arg.tanh());
    }
}

// Our NEON Quick GELU
fn gelu_quick_neon(x: &[f32], dst: &mut [f32]) {
    unsafe {
        infer_train::ops::gelu::gelu_quick_f32(x, dst);
    }
}

// llama.cpp-style scalar Quick GELU
fn gelu_quick_scalar(x: &[f32], dst: &mut [f32]) {
    const GELU_QUICK_COEF: f32 = -1.702;
    for i in 0..x.len() {
        dst[i] = x[i] / (1.0 + (GELU_QUICK_COEF * x[i]).exp());
    }
}

fn benchmark<F: Fn(&[f32], &mut [f32])>(name: &str, f: F, x: &[f32], iterations: usize) -> f64 {
    let mut dst = vec![0.0f32; x.len()];

    // Warmup
    for _ in 0..100 {
        f(x, &mut dst);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f(x, &mut dst);
    }
    let elapsed = start.elapsed().as_secs_f64();

    elapsed / iterations as f64
}

fn main() {
    println!("=== SiLU/GELU Benchmark: NEON vs Scalar ===\n");

    let iterations = 10000;
    let sizes = [256, 512, 1024, 2048, 4096, 8192];

    // Generate test data
    println!("SiLU Performance:");
    println!("{:8} | {:12} | {:12} | {:8}", "Size", "NEON (μs)", "Scalar (μs)", "Speedup");
    println!("{}", "-".repeat(50));

    for &size in &sizes {
        let x: Vec<f32> = (0..size).map(|i| ((i as f32 - size as f32 / 2.0) * 0.01).sin()).collect();

        let t_neon = benchmark("NEON", silu_neon, &x, iterations) * 1e6;
        let t_scalar = benchmark("Scalar", silu_scalar, &x, iterations) * 1e6;
        let speedup = t_scalar / t_neon;

        println!("{:8} | {:12.2} | {:12.2} | {:8.2}x", size, t_neon, t_scalar, speedup);
    }

    println!("\nGELU Performance:");
    println!("{:8} | {:12} | {:12} | {:8}", "Size", "NEON (μs)", "Scalar (μs)", "Speedup");
    println!("{}", "-".repeat(50));

    for &size in &sizes {
        let x: Vec<f32> = (0..size).map(|i| ((i as f32 - size as f32 / 2.0) * 0.01).sin()).collect();

        let t_neon = benchmark("NEON", gelu_neon, &x, iterations) * 1e6;
        let t_scalar = benchmark("Scalar", gelu_scalar, &x, iterations) * 1e6;
        let speedup = t_scalar / t_neon;

        println!("{:8} | {:12.2} | {:12.2} | {:8.2}x", size, t_neon, t_scalar, speedup);
    }

    println!("\nGELU-Quick Performance:");
    println!("{:8} | {:12} | {:12} | {:8}", "Size", "NEON (μs)", "Scalar (μs)", "Speedup");
    println!("{}", "-".repeat(50));

    for &size in &sizes {
        let x: Vec<f32> = (0..size).map(|i| ((i as f32 - size as f32 / 2.0) * 0.01).sin()).collect();

        let t_neon = benchmark("NEON", gelu_quick_neon, &x, iterations) * 1e6;
        let t_scalar = benchmark("Scalar", gelu_quick_scalar, &x, iterations) * 1e6;
        let speedup = t_scalar / t_neon;

        println!("{:8} | {:12.2} | {:12.2} | {:8.2}x", size, t_neon, t_scalar, speedup);
    }

    // Correctness check
    println!("\n=== Correctness Check ===");
    let test_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    let mut dst_neon = vec![0.0; test_data.len()];
    let mut dst_scalar = vec![0.0; test_data.len()];

    silu_neon(&test_data, &mut dst_neon);
    silu_scalar(&test_data, &mut dst_scalar);

    let mut max_diff = 0.0f32;
    for i in 0..test_data.len() {
        max_diff = max_diff.max((dst_neon[i] - dst_scalar[i]).abs());
    }
    println!("SiLU max diff: {}", max_diff);

    gelu_neon(&test_data, &mut dst_neon);
    gelu_scalar(&test_data, &mut dst_scalar);

    max_diff = 0.0;
    for i in 0..test_data.len() {
        max_diff = max_diff.max((dst_neon[i] - dst_scalar[i]).abs());
    }
    println!("GELU max diff: {}", max_diff);

    gelu_quick_neon(&test_data, &mut dst_neon);
    gelu_quick_scalar(&test_data, &mut dst_scalar);

    max_diff = 0.0;
    println!("\nGELU-Quick detailed:");
    for i in 0..test_data.len() {
        let diff = (dst_neon[i] - dst_scalar[i]).abs();
        max_diff = max_diff.max(diff);
        println!("  x={}: NEON={}, Scalar={}, diff={}", test_data[i], dst_neon[i], dst_scalar[i], diff);
    }
    println!("GELU-Quick max diff: {}", max_diff);
}
