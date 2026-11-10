// Benchmark our NEON vec_dot vs scalar implementation
// This tests if the vec_dot kernel itself is the bottleneck

use infer_train::quant::kv_cache::quantize_row_q8_k_neon;
use infer_train::quant::matmul::k_quant::matmul_q4_k_q8_k_single_thread;
use infer_train::quant::types::{BlockQ4K, BlockQ5K, BlockQ8K, QK_K};
use infer_train::quant::vec_dot::arm::k_quant::*;
use std::arch::aarch64::*;
use std::time::{Duration, Instant};

fn main() {
    println!("=== Vec_Dot Kernel Benchmark ===\n");

    // Test different sizes matching Qwen3-0.6B
    let k = 1024; // hidden_size
    let nb = k / QK_K;

    // Create test data
    let mut x_q8k: Vec<BlockQ8K> = vec![unsafe { std::mem::zeroed() }; nb];
    let mut weights_q4k: Vec<BlockQ4K> =
        vec![unsafe { std::mem::zeroed() }; nb];
    let mut weights_q5k: Vec<BlockQ5K> =
        vec![unsafe { std::mem::zeroed() }; nb];

    // Initialize with random values
    for i in 0..nb {
        x_q8k[i].d = 1.0;
        for j in 0..QK_K {
            x_q8k[i].qs[j] = (j % 256) as i8;
        }
        for j in 0..16 {
            x_q8k[i].bsums[j] = 100;
        }

        // Use raw u16 values for f16 (0.5 in f16 is 0x3800)
        weights_q4k[i].d = 0x3800; // 0.5 in f16
        weights_q4k[i].dmin = 0;
        for j in 0..128 {
            weights_q4k[i].qs[j] = (j % 256) as u8;
        }
        for j in 0..12 {
            weights_q4k[i].scales[j] = 10;
        }

        weights_q5k[i].d = 0x3800; // 0.5 in f16
        weights_q5k[i].dmin = 0;
        for j in 0..128 {
            weights_q5k[i].qs[j] = (j % 256) as u8;
        }
        for j in 0..12 {
            weights_q5k[i].scales[j] = 10;
        }
    }

    let iterations = 10000;
    let warmup = 100;

    // Test Q4_K vec_dot
    println!("Q4_K vec_dot (k=1024, {} iterations):", iterations);

    // Warmup
    for _ in 0..warmup {
        unsafe {
            let _ = vec_dot_q4_k_q8_k_neon(k, &weights_q4k, &x_q8k);
        }
    }

    // Benchmark NEON
    let mut total_time = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        unsafe {
            let _ = vec_dot_q4_k_q8_k_neon(k, &weights_q4k, &x_q8k);
        }
        total_time += start.elapsed();
    }
    let neon_avg_ns = total_time.as_nanos() as f64 / iterations as f64;
    let neon_gbs = (k as f64 * 4.0 / 1e9) / (neon_avg_ns / 1e9);

    println!("  NEON: {:.2} ns/call ({:.2} GB/s)", neon_avg_ns, neon_gbs);

    // Test Q5_K vec_dot
    println!("\nQ5_K vec_dot (k=1024, {} iterations):", iterations);

    // Warmup
    for _ in 0..warmup {
        unsafe {
            let _ = vec_dot_q5_k_q8_k_neon(k, &weights_q5k, &x_q8k);
        }
    }

    // Benchmark NEON
    let mut total_time = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        unsafe {
            let _ = vec_dot_q5_k_q8_k_neon(k, &weights_q5k, &x_q8k);
        }
        total_time += start.elapsed();
    }
    let neon_avg_ns = total_time.as_nanos() as f64 / iterations as f64;
    let neon_gbs = (k as f64 * 4.0 / 1e9) / (neon_avg_ns / 1e9);

    println!("  NEON: {:.2} ns/call ({:.2} GB/s)", neon_avg_ns, neon_gbs);

    // Test GEMV with multiple rows (matching decode workload)
    println!("\nGEMV Benchmark (m=2048 rows, k=1024):");

    let m = 2048; // FFN intermediate size
    let weights_q4k_2048: Vec<BlockQ4K> =
        vec![unsafe { std::mem::zeroed() }; m * nb];
    let weights_q5k_2048: Vec<BlockQ5K> =
        vec![unsafe { std::mem::zeroed() }; m * nb];
    let mut dst = vec![0.0f32; m];

    // Initialize weights
    for i in 0..m * nb {
        unsafe {
            let ptr = weights_q4k_2048.as_ptr().add(i) as *mut BlockQ4K;
            (*ptr).d = 0x3800; // 0.5 in f16
            (*ptr).dmin = 0;
        }
    }

    let gemv_iterations = 100;

    // Warmup
    for _ in 0..10 {
        unsafe {
            matmul_q4_k_q8_k_single_thread(
                &weights_q4k_2048,
                &x_q8k,
                &mut dst,
                k,
                m,
            );
        }
    }

    // Benchmark single-thread GEMV
    let mut total_time = Duration::ZERO;
    for _ in 0..gemv_iterations {
        let start = Instant::now();
        unsafe {
            matmul_q4_k_q8_k_single_thread(
                &weights_q4k_2048,
                &x_q8k,
                &mut dst,
                k,
                m,
            );
        }
        total_time += start.elapsed();
    }
    let gemv_avg_us = total_time.as_secs_f64() * 1e6 / gemv_iterations as f64;
    let gemv_gflops = (2.0 * m as f64 * k as f64 / gemv_avg_us / 1e6);

    println!(
        "  Q4_K single-thread GEMV: {:.2} µs ({:.2} GFLOPS)",
        gemv_avg_us, gemv_gflops
    );

    // Calculate how much time decode spends on GEMV
    println!("\n=== Decode Time Breakdown ===");
    let decode_time_ms = 22.46; // From benchmark
    let decode_layers = 28;

    // Estimate GEMV time
    // For Qwen3-0.6B:
    // - QKV: 3 * (n_heads + 2 * n_kv_heads) * hidden_size = 3 * (16 + 2*8) * 1024 = 98304 ops
    // - Wo: hidden_size * hidden_size = 1024 * 1024 = 1M ops
    // - FFN: 3 * intermediate_size * hidden_size = 3 * 2048 * 1024 = 6M ops
    // Total per layer: ~8M ops * 28 layers = 224M FLOPS

    let flops_per_layer = 8.0 * 1024.0 * 1024.0; // rough estimate
    let total_flops = flops_per_layer * decode_layers as f64;
    let gemv_time_estimate_ms = total_flops / (gemv_gflops * 1e6) * 1e3;

    println!("Estimated GEMV time: {:.2} ms", gemv_time_estimate_ms);
    println!("Non-GEMV time: {:.2} ms", decode_time_ms - gemv_time_estimate_ms);
}
