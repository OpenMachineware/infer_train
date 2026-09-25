// Benchmark RMSNorm: Metal vs CPU NEON vs llama.cpp scalar
// Run: cargo run --release --bin benchmark_rms_norm_metal

use infer_train::ops::rms_norm::rms_norm_f32;
use infer_train::ops::gpu::metal::RmsNormMetalContext;
use std::time::Instant;

/// llama.cpp scalar RMSNorm (from ggml-cpu/ops.cpp)
fn rms_norm_scalar_llama_cpp(
    x: &[f32],
    w: &[f32],
    dst: &mut [f32],
    hidden_dim: usize,
    eps: f32,
) {
    let n_rows = x.len() / hidden_dim;

    for row in 0..n_rows {
        let row_offset = row * hidden_dim;
        let x_row = &x[row_offset..row_offset + hidden_dim];
        let dst_row = &mut dst[row_offset..row_offset + hidden_dim];

        let mut sum: f64 = 0.0;
        for i in 0..hidden_dim {
            sum += (x_row[i] * x_row[i]) as f64;
        }

        let mean = sum / hidden_dim as f64;
        let scale = (1.0f64 / (mean + eps as f64).sqrt()) as f32;

        for i in 0..hidden_dim {
            dst_row[i] = x_row[i] * scale * w[i];
        }
    }
}

fn main() {
    println!("=== RMSNorm: Metal vs CPU NEON vs llama.cpp Scalar ===\n");

    let ctx = match RmsNormMetalContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to create Metal context: {}", e);
            return;
        }
    };

    let hidden_dim = 4096;
    let eps = 1e-5;

    // Test different batch sizes
    let batch_sizes = [1, 4, 16, 64, 256];

    println!("{:<10} {:<15} {:<15} {:<15} {:<10} {:<10} {:<10}",
             "Batch", "Metal (us)", "NEON (us)", "Scalar (us)", "M/N", "M/S", "N/S");
    println!("{}", "-".repeat(95));

    for batch in batch_sizes {
        let x: Vec<f32> = (0..batch * hidden_dim)
            .map(|i| (i % 100) as f32 * 0.01 - 0.5)
            .collect();
        let w: Vec<f32> = (0..hidden_dim).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect();

        let mut dst_neon = vec![0.0f32; batch * hidden_dim];
        let mut dst_scalar = vec![0.0f32; batch * hidden_dim];

        // Warmup
        for _ in 0..10 {
            let _ = ctx.rms_norm_f32(&x, &w, hidden_dim, eps);
            unsafe {
                rms_norm_f32(&x, &w, &mut dst_neon, hidden_dim, eps);
            }
            rms_norm_scalar_llama_cpp(&x, &w, &mut dst_scalar, hidden_dim, eps);
        }

        let iterations = (10000 / batch).max(100);

        // Benchmark Metal
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ctx.rms_norm_f32(&x, &w, hidden_dim, eps);
        }
        let metal_time = start.elapsed().as_micros() as f64 / iterations as f64;

        // Benchmark NEON
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                rms_norm_f32(&x, &w, &mut dst_neon, hidden_dim, eps);
            }
        }
        let neon_time = start.elapsed().as_micros() as f64 / iterations as f64;

        // Benchmark scalar
        let start = Instant::now();
        for _ in 0..iterations {
            rms_norm_scalar_llama_cpp(&x, &w, &mut dst_scalar, hidden_dim, eps);
        }
        let scalar_time = start.elapsed().as_micros() as f64 / iterations as f64;

        let metal_neon = neon_time / metal_time;
        let metal_scalar = scalar_time / metal_time;
        let neon_scalar = scalar_time / neon_time;

        println!(
            "{:<10} {:<15.2} {:<15.2} {:<15.2} {:<10.2}x {:<10.2}x {:<10.2}x",
            batch, metal_time, neon_time, scalar_time, metal_neon, metal_scalar, neon_scalar
        );

        // Verify correctness
        let dst_metal = ctx.rms_norm_f32(&x, &w, hidden_dim, eps).unwrap();
        let mut max_diff = 0.0f32;
        for i in 0..batch * hidden_dim {
            let diff = (dst_metal[i] - dst_neon[i]).abs();
            max_diff = max_diff.max(diff);
        }
        if max_diff > 1e-5 {
            println!("  WARNING: Metal vs NEON max_diff = {:.9}", max_diff);
        }
    }

    // Test different hidden dimensions
    println!("\n=== Hidden Dimension Scaling ===\n");
    let dims = [512, 1024, 2048, 4096, 8192];

    println!("{:<10} {:<15} {:<15} {:<15} {:<10} {:<10}",
             "Dim", "Metal (us)", "NEON (us)", "Scalar (us)", "M/N", "M/S");
    println!("{}", "-".repeat(75));

    for dim in dims {
        let x: Vec<f32> = (0..dim).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect();
        let mut dst_neon = vec![0.0f32; dim];
        let mut dst_scalar = vec![0.0f32; dim];

        // Warmup
        for _ in 0..10 {
            let _ = ctx.rms_norm_f32(&x, &w, dim, eps);
            unsafe {
                rms_norm_f32(&x, &w, &mut dst_neon, dim, eps);
            }
            rms_norm_scalar_llama_cpp(&x, &w, &mut dst_scalar, dim, eps);
        }

        let iterations = (100000 / (dim / 512)).max(1000);

        // Metal
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ctx.rms_norm_f32(&x, &w, dim, eps);
        }
        let metal_time = start.elapsed().as_micros() as f64 / iterations as f64;

        // NEON
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                rms_norm_f32(&x, &w, &mut dst_neon, dim, eps);
            }
        }
        let neon_time = start.elapsed().as_micros() as f64 / iterations as f64;

        // Scalar
        let start = Instant::now();
        for _ in 0..iterations {
            rms_norm_scalar_llama_cpp(&x, &w, &mut dst_scalar, dim, eps);
        }
        let scalar_time = start.elapsed().as_micros() as f64 / iterations as f64;

        let metal_neon = neon_time / metal_time;
        let metal_scalar = scalar_time / metal_time;

        println!(
            "{:<10} {:<15.2} {:<15.2} {:<15.2} {:<10.2}x {:<10.2}x",
            dim, metal_time, neon_time, scalar_time, metal_neon, metal_scalar
        );
    }

    // Calculate GFLOPS
    println!("\n=== GFLOPS ===\n");

    let dim = 4096;
    let x: Vec<f32> = (0..dim).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
    let w: Vec<f32> = (0..dim).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect();
    let mut dst_neon = vec![0.0f32; dim];
    let mut dst_scalar = vec![0.0f32; dim];

    // Warmup
    for _ in 0..100 {
        let _ = ctx.rms_norm_f32(&x, &w, dim, eps);
        unsafe {
            rms_norm_f32(&x, &w, &mut dst_neon, dim, eps);
        }
        rms_norm_scalar_llama_cpp(&x, &w, &mut dst_scalar, dim, eps);
    }

    let iterations = 10000;

    // Metal
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ctx.rms_norm_f32(&x, &w, dim, eps);
    }
    let metal_time = start.elapsed().as_secs_f64();

    // NEON
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            rms_norm_f32(&x, &w, &mut dst_neon, dim, eps);
        }
    }
    let neon_time = start.elapsed().as_secs_f64();

    // Scalar
    let start = Instant::now();
    for _ in 0..iterations {
        rms_norm_scalar_llama_cpp(&x, &w, &mut dst_scalar, dim, eps);
    }
    let scalar_time = start.elapsed().as_secs_f64();

    let flops = iterations as f64 * dim as f64 * 3.0;
    let metal_gflops = flops / metal_time / 1e9;
    let neon_gflops = flops / neon_time / 1e9;
    let scalar_gflops = flops / scalar_time / 1e9;

    println!("Metal:    {:>8.2} GFLOPS", metal_gflops);
    println!("NEON:     {:>8.2} GFLOPS", neon_gflops);
    println!("Scalar:   {:>8.2} GFLOPS", scalar_gflops);
    println!();
    println!("Metal vs NEON:   {:>6.2}x", metal_gflops / neon_gflops);
    println!("Metal vs Scalar: {:>6.2}x", metal_gflops / scalar_gflops);
}
