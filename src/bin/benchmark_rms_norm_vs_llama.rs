// Compare RMSNorm performance: our NEON vs llama.cpp scalar
// Run: cargo run --release --bin benchmark_rms_norm_vs_llama

use infer_train::ops::rms_norm::rms_norm_f32;
use std::time::Instant;

/// llama.cpp scalar RMSNorm (from ggml-cpu/ops.cpp)
/// Exact copy of their implementation for fair comparison
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

        // llama.cpp: sum += (ggml_float)(x[i00] * x[i00]);
        let mut sum: f64 = 0.0;
        for i in 0..hidden_dim {
            sum += (x_row[i] * x_row[i]) as f64;
        }

        // llama.cpp: const float mean = sum/ne00;
        // llama.cpp: const float scale = 1.0f/sqrtf(mean + eps);
        let mean = sum / hidden_dim as f64;
        let scale = (1.0f64 / (mean + eps as f64).sqrt()) as f32;

        // llama.cpp: y[i00] = x[i00]*scale*w[i00];
        for i in 0..hidden_dim {
            dst_row[i] = x_row[i] * scale * w[i];
        }
    }
}

fn main() {
    println!("=== RMSNorm: NEON vs llama.cpp Scalar ===\n");

    let hidden_dim = 4096;
    let eps = 1e-5;

    // Test different batch sizes
    let batch_sizes = [1, 4, 16, 64, 256];

    println!("{:<10} {:<15} {:<15} {:<10}", "Batch", "NEON (us)", "Scalar (us)", "Speedup");
    println!("{}", "-".repeat(50));

    for batch in batch_sizes {
        let x: Vec<f32> = (0..batch * hidden_dim)
            .map(|i| (i % 100) as f32 * 0.01 - 0.5)
            .collect();
        let w: Vec<f32> = (0..hidden_dim).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect();

        // Warmup and benchmark NEON
        let mut dst_neon = vec![0.0f32; batch * hidden_dim];
        for _ in 0..100 {
            unsafe {
                rms_norm_f32(&x, &w, &mut dst_neon, hidden_dim, eps);
            }
        }

        let iterations = (10000 / batch).max(100);
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                rms_norm_f32(&x, &w, &mut dst_neon, hidden_dim, eps);
            }
        }
        let neon_time = start.elapsed().as_micros() as f64 / iterations as f64;

        // Warmup and benchmark scalar
        let mut dst_scalar = vec![0.0f32; batch * hidden_dim];
        for _ in 0..100 {
            rms_norm_scalar_llama_cpp(&x, &w, &mut dst_scalar, hidden_dim, eps);
        }

        let start = Instant::now();
        for _ in 0..iterations {
            rms_norm_scalar_llama_cpp(&x, &w, &mut dst_scalar, hidden_dim, eps);
        }
        let scalar_time = start.elapsed().as_micros() as f64 / iterations as f64;

        let speedup = scalar_time / neon_time;

        println!(
            "{:<10} {:<15.2} {:<15.2} {:<10.2}x",
            batch, neon_time, scalar_time, speedup
        );

        // Verify correctness
        let mut max_diff = 0.0f32;
        let mut max_idx = 0;
        for i in 0..batch * hidden_dim {
            let diff = (dst_neon[i] - dst_scalar[i]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
            }
        }
        if max_diff > 1e-6 {
            println!("  WARNING: max_diff = {:.9} at idx {}", max_diff, max_idx);
            println!("  dst_neon[{}] = {:.9}, dst_scalar[{}] = {:.9}", max_idx, dst_neon[max_idx], max_idx, dst_scalar[max_idx]);
        }
    }

    // Test different hidden dimensions
    println!("\n=== Hidden Dimension Scaling ===\n");
    let dims = [512, 1024, 2048, 4096, 8192];

    println!("{:<10} {:<15} {:<15} {:<10}", "Dim", "NEON (us)", "Scalar (us)", "Speedup");
    println!("{}", "-".repeat(50));

    for dim in dims {
        let x: Vec<f32> = (0..dim).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
        let w: Vec<f32> = (0..dim).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect();
        let mut dst_neon = vec![0.0f32; dim];
        let mut dst_scalar = vec![0.0f32; dim];

        // Warmup
        for _ in 0..100 {
            unsafe {
                rms_norm_f32(&x, &w, &mut dst_neon, dim, eps);
            }
            rms_norm_scalar_llama_cpp(&x, &w, &mut dst_scalar, dim, eps);
        }

        let iterations = (100000 / (dim / 512)).max(1000);

        // Benchmark NEON
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                rms_norm_f32(&x, &w, &mut dst_neon, dim, eps);
            }
        }
        let neon_time = start.elapsed().as_micros() as f64 / iterations as f64;

        // Benchmark scalar
        let start = Instant::now();
        for _ in 0..iterations {
            rms_norm_scalar_llama_cpp(&x, &w, &mut dst_scalar, dim, eps);
        }
        let scalar_time = start.elapsed().as_micros() as f64 / iterations as f64;

        let speedup = scalar_time / neon_time;

        println!(
            "{:<10} {:<15.2} {:<15.2} {:<10.2}x",
            dim, neon_time, scalar_time, speedup
        );
    }

    // Calculate GFLOPS
    println!("\n=== GFLOPS Comparison ===\n");

    let dim = 4096;
    let x: Vec<f32> = (0..dim).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
    let w: Vec<f32> = (0..dim).map(|i| 1.0 + (i % 10) as f32 * 0.1).collect();
    let mut dst_neon = vec![0.0f32; dim];
    let mut dst_scalar = vec![0.0f32; dim];

    // Warmup
    for _ in 0..100 {
        unsafe {
            rms_norm_f32(&x, &w, &mut dst_neon, dim, eps);
        }
        rms_norm_scalar_llama_cpp(&x, &w, &mut dst_scalar, dim, eps);
    }

    let iterations = 10000;

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

    let flops = iterations as f64 * dim as f64 * 3.0; // ~3 FLOPs per element
    let neon_gflops = flops / neon_time / 1e9;
    let scalar_gflops = flops / scalar_time / 1e9;

    println!("NEON:   {:.2} GFLOPS", neon_gflops);
    println!("Scalar: {:.2} GFLOPS", scalar_gflops);
    println!("Speedup: {:.2}x", neon_gflops / scalar_gflops);
}
