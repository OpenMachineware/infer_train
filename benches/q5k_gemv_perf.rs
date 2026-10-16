// Q5_K GEMV Performance Benchmark
// Compare our implementation against llama.cpp baseline

use bytemuck::Zeroable;
use half::f16;
use infer_train::inference::dequant::dequant_q5_k;
use infer_train::quant::types::{BlockQ5K, QK_K};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::*;
use std::time::Instant;

fn main() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    // Test different matrix sizes
    let test_cases = vec![
        (128, 4096),  // Small
        (512, 4096),  // Medium
        (1024, 4096), // Large
        (2048, 4096), // Very large
    ];

    println!("Q5_K GEMV Performance Benchmark");
    println!("================================");
    println!("Comparing GPU (Metal) vs CPU (Rust) implementations\n");

    // First: verify correctness
    println!("=== Correctness Check ===\n");
    let (m, k) = (256, 4096);
    let n_blocks = k / QK_K;
    let mut weights = vec![BlockQ5K::zeroed(); m * n_blocks];
    for (idx, block) in weights.iter_mut().enumerate() {
        block.d = f16::to_bits(f16::from_f32(0.5 + 0.001 * (idx % 100) as f32));
        block.dmin = f16::to_bits(f16::from_f32(0.1 * ((idx % 10) as f32)));
        for j in 0..12 {
            block.scales[j] = ((idx * 7 + j * 11 + 1) % 64) as u8;
        }
        for j in 0..32 {
            block.qh[j] = ((idx * 13 + j * 17) % 256) as u8;
        }
        for j in 0..128 {
            block.qs[j] = ((idx * 13 + j * 17 + 8) % 256) as u8;
        }
    }
    let input: Vec<f32> =
        (0..k).map(|i| 0.1 * ((i % 10) as f32 + 1.0)).collect();

    // GPU computation
    let weights_buffer = ctx.device().new_buffer(
        (weights.len() * std::mem::size_of::<BlockQ5K>()) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            weights.as_ptr() as *const u8,
            weights_buffer.contents() as *mut u8,
            weights.len() * std::mem::size_of::<BlockQ5K>(),
        );
    }
    weights_buffer.did_modify_range(NSRange {
        location: 0,
        length: weights_buffer.length(),
    });
    let input_buffer = ctx.device().new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        (input.len() * 4) as u64,
        MTLResourceOptions::StorageModeManaged,
    );
    let output_buffer = ctx
        .mv_q5_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
        .expect("Q5_K MV failed");
    let gpu_output = unsafe {
        std::slice::from_raw_parts(output_buffer.contents() as *const f32, m)
    };

    // CPU reference
    let mut cpu_output = vec![0.0f32; m];
    for row in 0..m {
        let mut sum = 0.0f32;
        for block_idx in 0..n_blocks {
            let block = &weights[row * n_blocks + block_idx];
            let block_bytes = unsafe {
                std::slice::from_raw_parts(
                    block as *const BlockQ5K as *const u8,
                    std::mem::size_of::<BlockQ5K>(),
                )
            };
            let mut dequant_values = [0.0f32; QK_K];
            dequant_q5_k(block_bytes, &mut dequant_values, QK_K);
            for i in 0..QK_K {
                sum += dequant_values[i] * input[block_idx * QK_K + i];
            }
        }
        cpu_output[row] = sum;
    }

    // Compare
    let mut max_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut mismatches = 0;
    let mut mismatch_rows = Vec::new();
    for i in 0..m {
        let diff = (gpu_output[i] - cpu_output[i]).abs();
        let rel_diff = if cpu_output[i].abs() > 1.0 {
            diff / cpu_output[i].abs()
        } else {
            diff
        };
        if diff > max_diff {
            max_diff = diff;
        }
        if rel_diff > max_rel_diff {
            max_rel_diff = rel_diff;
        }
        // For large values, use relative error (1% tolerance for FP accumulation order diff)
        // For small values, use absolute error
        let is_large_value = cpu_output[i].abs() > 1000.0;
        let is_error_acceptable = if is_large_value {
            rel_diff < 0.01 // 1% relative error for large values
        } else {
            diff < 2.0 // 2.0 absolute error for small values
        };
        if !is_error_acceptable {
            mismatches += 1;
            mismatch_rows.push(i);
        }
    }
    println!("Matrix: {} × {} ({} blocks)", m, k, n_blocks);
    println!("  max_diff: {:.6}", max_diff);
    println!(
        "  max_rel_diff: {:.6}% ({:.6})",
        max_rel_diff * 100.0,
        max_rel_diff
    );
    println!("  mismatches: {} / {}", mismatches, m);
    if mismatches > 0 {
        println!(
            "  First 10 mismatch rows: {:?}",
            &mismatch_rows[..mismatch_rows.len().min(10)]
        );
        for &row in mismatch_rows.iter().take(3) {
            println!(
                "    Row {}: GPU={:.6}, CPU={:.6}, diff={:.6}",
                row,
                gpu_output[row],
                cpu_output[row],
                (gpu_output[row] - cpu_output[row]).abs()
            );
        }
    }
    if mismatches == 0 {
        println!("  ✓ Correctness verified\n");
    } else {
        println!("  ✗ Correctness needs investigation\n");
    }
    println!("=== Performance Benchmark ===\n");

    for (m, k) in test_cases {
        let n_blocks = k / QK_K;
        let n_iterations = 100;

        // Create test data
        let mut weights = vec![BlockQ5K::zeroed(); m * n_blocks];
        for (idx, block) in weights.iter_mut().enumerate() {
            block.d =
                f16::to_bits(f16::from_f32(0.5 + 0.001 * (idx % 100) as f32));
            block.dmin = f16::to_bits(f16::from_f32(0.1 * ((idx % 10) as f32)));
            for j in 0..12 {
                block.scales[j] = ((idx * 7 + j * 11 + 1) % 64) as u8;
            }
            for j in 0..32 {
                block.qh[j] = ((idx * 13 + j * 17) % 256) as u8;
            }
            for j in 0..128 {
                block.qs[j] = ((idx * 13 + j * 17 + 8) % 256) as u8;
            }
        }

        let input: Vec<f32> =
            (0..k).map(|i| 0.1 * ((i % 10) as f32 + 1.0)).collect();

        // Create GPU buffers
        let weights_buffer = ctx.device().new_buffer(
            (weights.len() * std::mem::size_of::<BlockQ5K>()) as u64,
            MTLResourceOptions::StorageModeManaged,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                weights.as_ptr() as *const u8,
                weights_buffer.contents() as *mut u8,
                weights.len() * std::mem::size_of::<BlockQ5K>(),
            );
        }
        weights_buffer.did_modify_range(NSRange {
            location: 0,
            length: weights_buffer.length(),
        });

        let input_buffer = ctx.device().new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (input.len() * 4) as u64,
            MTLResourceOptions::StorageModeManaged,
        );

        // Warm up GPU
        let _ =
            ctx.mv_q5_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer);

        // Benchmark GPU
        let gpu_start = Instant::now();
        for _ in 0..n_iterations {
            let output_buffer = ctx
                .mv_q5_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
                .expect("Q5_K MV failed");
            // Force synchronization
            output_buffer.contents();
        }
        let gpu_time = gpu_start.elapsed();
        let gpu_avg_us = gpu_time.as_micros() as f64 / n_iterations as f64;

        // Benchmark CPU (single iteration for comparison, CPU is much slower)
        let cpu_start = Instant::now();
        let mut cpu_output = vec![0.0f32; m];
        for row in 0..m {
            let mut sum = 0.0f32;
            for block_idx in 0..n_blocks {
                let block = &weights[row * n_blocks + block_idx];
                let block_bytes = unsafe {
                    std::slice::from_raw_parts(
                        block as *const BlockQ5K as *const u8,
                        std::mem::size_of::<BlockQ5K>(),
                    )
                };
                let mut dequant_values = [0.0f32; QK_K];
                dequant_q5_k(block_bytes, &mut dequant_values, QK_K);
                for i in 0..QK_K {
                    sum += dequant_values[i] * input[block_idx * QK_K + i];
                }
            }
            cpu_output[row] = sum;
        }
        let cpu_time = cpu_start.elapsed();
        let cpu_avg_us = cpu_time.as_micros() as f64;

        // Calculate operations
        let ops = m as f64 * k as f64 * 2.0; // multiply-add = 2 ops
        let gpu_gops = ops / (gpu_avg_us * 1e-6) / 1e9;
        let cpu_gops = ops / (cpu_avg_us * 1e-6) / 1e9;
        let speedup = cpu_avg_us / gpu_avg_us;

        println!("Matrix: {} × {} ({} blocks)", m, k, n_blocks);
        println!("  GPU: {:8.2} µs/iter ({:6.2} GOPS)", gpu_avg_us, gpu_gops);
        println!("  CPU: {:8.2} µs/iter ({:6.2} GOPS)", cpu_avg_us, cpu_gops);
        println!("  Speedup: {:.2}x", speedup);
        println!();
    }

    // Baseline comparison (llama.cpp typical performance on M-series)
    // llama.cpp achieves ~40-80 GOPS for Q5_K GEMV on M1/M2
    // Our target: match or exceed llama.cpp
    println!(
        "Target: Match llama.cpp Q5_K performance (~40-80 GOPS on M-series)"
    );
}
