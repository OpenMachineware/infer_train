// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark binary ops (ADD, MUL, SUB, DIV) vs vDSP (Accelerate)
// llama.cpp uses vDSP for F32 binary ops on macOS

use half::f16;
use infer_train::ops::{
    add_f16, add_f32, div_f16, div_f32, mul_f16, mul_f32, sub_f16, sub_f32,
};
use std::time::Instant;

// vDSP types
#[cfg(target_os = "macos")]
type vDSP_Stride = std::os::raw::c_int;
#[cfg(target_os = "macos")]
type vDSP_Length = std::os::raw::c_uint;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vDSP_vadd(
        a: *const f32,
        ia: vDSP_Stride,
        b: *const f32,
        ib: vDSP_Stride,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    fn vDSP_vsub(
        a: *const f32,
        ia: vDSP_Stride,
        b: *const f32,
        ib: vDSP_Stride,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    fn vDSP_vmul(
        a: *const f32,
        ia: vDSP_Stride,
        b: *const f32,
        ib: vDSP_Stride,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
    fn vDSP_vdiv(
        a: *const f32,
        ia: vDSP_Stride,
        b: *const f32,
        ib: vDSP_Stride,
        c: *mut f32,
        ic: vDSP_Stride,
        n: vDSP_Length,
    );
}

#[cfg(target_os = "macos")]
fn benchmark_vdsp_f32(
    op: unsafe extern "C" fn(
        *const f32,
        vDSP_Stride,
        *const f32,
        vDSP_Stride,
        *mut f32,
        vDSP_Stride,
        vDSP_Length,
    ),
    n: usize,
    iterations: usize,
) -> f64 {
    let src0: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let src1: Vec<f32> = (0..n).map(|i| (i as f32 + 2.0) * 0.1).collect();
    let mut dst = vec![0.0f32; n];

    // Warmup
    for _ in 0..10 {
        unsafe {
            op(
                src1.as_ptr(),
                1,
                src0.as_ptr(),
                1,
                dst.as_mut_ptr(),
                1,
                n as vDSP_Length,
            );
        }
    }

    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            op(
                src1.as_ptr(),
                1,
                src0.as_ptr(),
                1,
                dst.as_mut_ptr(),
                1,
                n as vDSP_Length,
            );
        }
    }
    let elapsed = start.elapsed().as_secs_f64();

    (elapsed / iterations as f64) * 1_000_000.0
}

fn benchmark_ours_f32(
    op: fn(&[f32], &[f32], &mut [f32]),
    n: usize,
    iterations: usize,
) -> f64 {
    let src0: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let src1: Vec<f32> = (0..n).map(|i| (i as f32 + 2.0) * 0.1).collect();
    let mut dst = vec![0.0f32; n];

    // Warmup
    for _ in 0..10 {
        op(&src0, &src1, &mut dst);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        op(&src0, &src1, &mut dst);
    }
    let elapsed = start.elapsed().as_secs_f64();

    (elapsed / iterations as f64) * 1_000_000.0
}

fn benchmark_ours_f16(
    op: fn(&[f16], &[f16], &mut [f16]),
    n: usize,
    iterations: usize,
) -> f64 {
    let src0: Vec<f16> =
        (0..n).map(|i| f16::from_f32((i as f32 + 1.0) * 0.1)).collect();
    let src1: Vec<f16> =
        (0..n).map(|i| f16::from_f32((i as f32 + 2.0) * 0.1)).collect();
    let mut dst = vec![f16::ZERO; n];

    // Warmup
    for _ in 0..10 {
        op(&src0, &src1, &mut dst);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        op(&src0, &src1, &mut dst);
    }
    let elapsed = start.elapsed().as_secs_f64();

    (elapsed / iterations as f64) * 1_000_000.0
}

// Scalar reference implementation (matching llama.cpp's fallback)
fn scalar_add_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
    for i in 0..dst.len().min(src0.len()).min(src1.len()) {
        dst[i] = src0[i] + src1[i];
    }
}

fn scalar_mul_f32(src0: &[f32], src1: &[f32], dst: &mut [f32]) {
    for i in 0..dst.len().min(src0.len()).min(src1.len()) {
        dst[i] = src0[i] * src1[i];
    }
}

fn benchmark_scalar_f32(
    op: fn(&[f32], &[f32], &mut [f32]),
    n: usize,
    iterations: usize,
) -> f64 {
    let src0: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let src1: Vec<f32> = (0..n).map(|i| (i as f32 + 2.0) * 0.1).collect();
    let mut dst = vec![0.0f32; n];

    // Warmup
    for _ in 0..10 {
        op(&src0, &src1, &mut dst);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        op(&src0, &src1, &mut dst);
    }
    let elapsed = start.elapsed().as_secs_f64();

    (elapsed / iterations as f64) * 1_000_000.0
}

fn main() {
    println!(
        "Binary Operations Benchmark: Rust NEON vs vDSP (llama.cpp uses vDSP)\n"
    );

    let sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192];
    let iterations = [10000, 5000, 2000, 1000, 500, 200, 100, 50];

    #[cfg(target_os = "macos")]
    {
        println!("=== F32: NEON vs vDSP ===\n");
        println!(
            "Size     | NEON ADD | vDSP ADD | Ratio  | NEON MUL | vDSP MUL | Ratio  | NEON SUB | vDSP SUB | Ratio  | NEON DIV | vDSP DIV | Ratio"
        );
        println!(
            "---------|----------|----------|--------|----------|----------|--------|----------|----------|--------|----------|----------|-------"
        );

        for (i, &n) in sizes.iter().enumerate() {
            let iters = iterations[i];

            // ADD
            let our_add = benchmark_ours_f32(add_f32, n, iters);
            let vdsp_add = benchmark_vdsp_f32(vDSP_vadd, n, iters);
            let add_ratio = our_add / vdsp_add * 100.0;

            // MUL
            let our_mul = benchmark_ours_f32(mul_f32, n, iters);
            let vdsp_mul = benchmark_vdsp_f32(vDSP_vmul, n, iters);
            let mul_ratio = our_mul / vdsp_mul * 100.0;

            // SUB (vDSP uses opposite order: vDSP_vsub(b, a) = a - b)
            let our_sub = benchmark_ours_f32(sub_f32, n, iters);
            let vdsp_sub = benchmark_vdsp_f32(vDSP_vsub, n, iters);
            let sub_ratio = our_sub / vdsp_sub * 100.0;

            // DIV
            let our_div = benchmark_ours_f32(div_f32, n, iters);
            let vdsp_div = benchmark_vdsp_f32(vDSP_vdiv, n, iters);
            let div_ratio = our_div / vdsp_div * 100.0;

            println!(
                "{:8} | {:8.2} | {:8.2} | {:5.1}% | {:8.2} | {:8.2} | {:5.1}% | {:8.2} | {:8.2} | {:5.1}% | {:8.2} | {:8.2} | {:5.1}%",
                n,
                our_add,
                vdsp_add,
                add_ratio,
                our_mul,
                vdsp_mul,
                mul_ratio,
                our_sub,
                vdsp_sub,
                sub_ratio,
                our_div,
                vdsp_div,
                div_ratio
            );
        }

        println!("\n=== F16: Scalar (same as llama.cpp) ===\n");
        println!("Size     | ADD (us) | MUL (us) | SUB (us) | DIV (us)");
        println!("---------|----------|----------|----------|----------");

        for (i, &n) in sizes.iter().enumerate() {
            let iters = iterations[i];

            let add_f16_us = benchmark_ours_f16(add_f16, n, iters);
            let mul_f16_us = benchmark_ours_f16(mul_f16, n, iters);
            let sub_f16_us = benchmark_ours_f16(sub_f16, n, iters);
            let div_f16_us = benchmark_ours_f16(div_f16, n, iters);

            println!(
                "{:8} | {:8.2} | {:8.2} | {:8.2} | {:8.2}",
                n, add_f16_us, mul_f16_us, sub_f16_us, div_f16_us
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        println!("=== F32: NEON vs Scalar ===\n");
        println!(
            "Size     | NEON ADD | Scalar ADD | Speedup | NEON MUL | Scalar MUL | Speedup"
        );
        println!(
            "---------|----------|------------|---------|----------|------------|--------"
        );

        for (i, &n) in sizes.iter().enumerate() {
            let iters = iterations[i];

            let neon_add = benchmark_ours_f32(add_f32, n, iters);
            let scalar_add = benchmark_scalar_f32(scalar_add_f32, n, iters);
            let neon_mul = benchmark_ours_f32(mul_f32, n, iters);
            let scalar_mul = benchmark_scalar_f32(scalar_mul_f32, n, iters);

            println!(
                "{:8} | {:8.2} | {:10.2} | {:5.2}x | {:8.2} | {:10.2} | {:5.2}x",
                n,
                neon_add,
                scalar_add,
                scalar_add / neon_add,
                neon_mul,
                scalar_mul,
                scalar_mul / neon_mul
            );
        }
    }
}
