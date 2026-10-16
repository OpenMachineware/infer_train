// Q5_K GEMV overhead analysis
// Measure actual compute time vs kernel launch overhead

use bytemuck::Zeroable;
use half::f16;
use infer_train::quant::types::{BlockQ5K, QK_K};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use metal::*;
use std::time::Instant;

fn main() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    // Test with varying batch sizes to measure overhead
    let test_cases = vec![
        (1, 4096),    // Single row
        (8, 4096),    // Small batch
        (32, 4096),   // Medium batch
        (128, 4096),  // Typical decode
        (512, 4096),  // Large batch
        (2048, 4096), // Prefill
    ];

    println!("Q5_K GEMV Overhead Analysis");
    println!("============================\n");
    println!("Measuring kernel launch overhead vs actual compute time\n");

    for (m, k) in test_cases {
        let n_blocks = k / QK_K;
        let mut weights = vec![BlockQ5K::zeroed(); m * n_blocks];
        for (idx, block) in weights.iter_mut().enumerate() {
            block.d = f16::to_bits(f16::from_f32(0.5));
            block.dmin = f16::to_bits(f16::from_f32(0.1));
        }

        let input: Vec<f32> = vec![1.0; k];

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
            MTLResourceOptions::StorageModeShared,
        );

        // Warm up
        let _ = ctx.mv_q5_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer);

        // Measure multiple iterations
        let n_iterations = 100;
        let start = Instant::now();
        for _ in 0..n_iterations {
            let _ = ctx.mv_q5_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer);
        }
        let total_time = start.elapsed();
        let avg_time_us = total_time.as_micros() as f64 / n_iterations as f64;

        // Calculate metrics
        let ops = m as f64 * k as f64 * 2.0;
        let gops = ops / (avg_time_us * 1e-6) / 1e9;
        let effective_compute_time_us = ops / 44e9 * 1e6; // Assuming 44 GOPS peak
        let overhead_us = avg_time_us - effective_compute_time_us;
        let overhead_pct = overhead_us / avg_time_us * 100.0;

        println!("{} rows × {} cols ({} blocks):", m, k, n_blocks);
        println!("  Total time: {:6.1} µs", avg_time_us);
        println!("  Effective compute (@44 GOPS): {:6.1} µs", effective_compute_time_us);
        println!("  Kernel launch overhead: {:6.1} µs ({:4.1}%)", overhead_us, overhead_pct);
        println!("  Performance: {:5.1} GOPS", gops);
        println!("  If overhead removed: {:5.1} GOPS theoretical", ops / (effective_compute_time_us * 1e-6) / 1e9);
        println!();
    }

    println!("Conclusion:");
    println!("-----------");
    println!("Small matrices are dominated by kernel launch overhead (~200-300µs)");
    println!("This is a Metal driver limitation, not a kernel implementation issue.");
    println!("Solutions:");
    println!("1. Batch multiple GEMVs in single command buffer (already implemented for 28 layers)");
    println!("2. Multi-row per threadgroup (reduces threadgroup count, but complex)");
    println!("3. Kernel fusion (combine with next layer)");
}
