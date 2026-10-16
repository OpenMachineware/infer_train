// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// F16 GEMV roofline analysis - theoretical limits

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== F16 GEMV Roofline Analysis ===\n");

        let ctx = MetalContext::new().expect("Failed to create MetalContext");
        let device = ctx.device();

        // Get GPU info
        println!("GPU: {}", device.name());
        println!(
            "Recommended max working set: {} MB",
            device.recommended_max_working_set_size() / 1024 / 1024
        );

        // Test size: 151936×1536 (Qwen 1.5B output layer)
        let m = 151936;
        let k = 1536;

        println!("\nTest size: {}×{}", m, k);
        println!("Total weights: {} MB", (m * k * 2) / 1024 / 1024);

        // Create test data
        let total_weights = m * k;
        let mut weights_f16 = Vec::with_capacity(total_weights);
        for i in 0..total_weights {
            let f32_val = ((i as f32 * 0.12345) % 2.0) - 1.0;
            let f16_val = half::f16::from_f32(f32_val);
            weights_f16.push(f16_val.to_bits());
        }

        let mut input_f32 = Vec::with_capacity(k);
        for i in 0..k {
            input_f32.push(((i as f32 * 0.98765) % 2.0) - 1.0);
        }

        // Create GPU buffers
        let weights_buffer = std::rc::Rc::new(device.new_buffer_with_data(
            weights_f16.as_ptr() as *const std::ffi::c_void,
            (total_weights * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let input_buffer = std::rc::Rc::new(device.new_buffer_with_data(
            input_f32.as_ptr() as *const std::ffi::c_void,
            (k * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Warmup
        for _ in 0..10 {
            let _ = ctx.mv_f16_f32_gpu_weights(
                m,
                k,
                &weights_buffer,
                &input_buffer,
            );
        }

        // Detailed benchmark
        println!("\nRunning 100 iterations...");
        let n_iters = 100;
        let start = std::time::Instant::now();

        for _ in 0..n_iters {
            let _ = ctx.mv_f16_f32_gpu_weights(
                m,
                k,
                &weights_buffer,
                &input_buffer,
            );
        }

        let total_time = start.elapsed();
        let avg_time = total_time.as_secs_f64() / n_iters as f64;

        // Theoretical analysis
        println!("\n=== Performance Analysis ===");

        // Operations
        let flops = 2.0 * m as f64 * k as f64;
        let gflops = flops / (avg_time * 1e9);
        println!("Operations: {:.2e} FLOPs", flops);
        println!("Time: {:.3} ms", avg_time * 1000.0);
        println!("Performance: {:.2} GFLOPS", gflops);

        // Memory
        let bytes_weights = (m * k * 2) as f64;
        let bytes_input = (k * 4) as f64;
        let bytes_output = (m * 4) as f64;
        let total_bytes = bytes_weights + bytes_input + bytes_output;

        let bandwidth_gbs = total_bytes / (avg_time * 1e9);

        println!("\nMemory traffic:");
        println!("  Weights: {:.2} MB", bytes_weights / 1e6);
        println!("  Input:   {:.2} MB", bytes_input / 1e6);
        println!("  Output:  {:.2} MB", bytes_output / 1e6);
        println!("  Total:   {:.2} MB", total_bytes / 1e6);
        println!("  Bandwidth: {:.2} GB/s", bandwidth_gbs);

        // Operational intensity
        let intensity = flops / total_bytes;
        println!("\nOperational intensity: {:.2} FLOPs/byte", intensity);
        println!("Arithmetic intensity: {:.2} FLOPs/byte", intensity);

        // Roofline analysis
        // Apple M2 GPU: ~500 GFLOPS (FP16), ~100 GB/s bandwidth
        // For intensity < 5, we're memory bound
        // For intensity > 5, we're compute bound

        let assumed_peak_gflops: f64 = 500.0;
        let assumed_bandwidth_gbs: f64 = 100.0;

        let roofline_peak =
            assumed_peak_gflops.min(intensity * assumed_bandwidth_gbs);
        let efficiency = gflops / roofline_peak * 100.0;

        println!("\n=== Roofline Model ===");
        println!("Assumed peak: {} GFLOPS", assumed_peak_gflops);
        println!("Assumed bandwidth: {} GB/s", assumed_bandwidth_gbs);
        println!("Roofline peak: {:.2} GFLOPS", roofline_peak);
        println!("Efficiency: {:.1}%", efficiency);

        // Memory bound analysis
        if intensity < 5.0 {
            println!("\nBound: MEMORY (intensity < 5)");
            println!("Maximum achievable: {:.2} GB/s", assumed_bandwidth_gbs);
            println!("Actual bandwidth: {:.2} GB/s", bandwidth_gbs);
            println!(
                "Bandwidth efficiency: {:.1}%",
                bandwidth_gbs / assumed_bandwidth_gbs * 100.0
            );
        } else {
            println!("\nBound: COMPUTE (intensity > 5)");
        }

        // Comparison with theoretical minimum
        let min_time_memory = total_bytes / (assumed_bandwidth_gbs * 1e9);
        let min_time_compute = flops / (assumed_peak_gflops * 1e9);

        println!("\nTheoretical minimum time:");
        println!("  Memory bound: {:.3} ms", min_time_memory * 1000.0);
        println!("  Compute bound: {:.3} ms", min_time_compute * 1000.0);
        println!("  Actual: {:.3} ms", avg_time * 1000.0);

        println!("\n=== Summary ===");
        println!("✓ Performance: {:.2} GFLOPS", gflops);
        println!("✓ Bandwidth: {:.2} GB/s", bandwidth_gbs);
        if efficiency > 50.0 {
            println!("✓ Efficiency: {:.1}% (good)", efficiency);
        } else {
            println!("⚠ Efficiency: {:.1}% (needs optimization)", efficiency);
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!(
            "F16 GEMV roofline analysis requires Metal support (macOS ARM64)"
        );
    }
}
