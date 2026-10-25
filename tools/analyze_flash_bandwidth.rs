// Analyze Flash Attention bandwidth utilization
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use std::rc::Rc;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx = MetalContext::new().expect("Failed to create Metal context");
        let device = ctx.device();
        let queue = ctx.queue();

        // Get device info
        println!("Metal Device: {}", device.name());
        println!(
            "Recommended Max Working Set: {} MB",
            device.recommended_max_working_set_size() / 1024 / 1024
        );
        println!();

        // Test parameters matching Qwen3-0.6B
        let n_heads = 32;
        let n_kv_heads = 8;
        let head_dim = 128;
        let max_seq = 2048;

        // Test different sequence lengths
        let seq_lengths = vec![128, 256, 512, 1024, 2048];

        println!("Flash Attention Bandwidth Analysis");
        println!(
            "Config: n_heads={}, n_kv_heads={}, head_dim={}",
            n_heads, n_kv_heads, head_dim
        );
        println!();

        // Create buffers
        let q_size = n_heads * head_dim * std::mem::size_of::<f32>();
        let kv_size =
            n_kv_heads * max_seq * head_dim * std::mem::size_of::<half::f16>();

        let q_data: Vec<f32> = (0..(n_heads * head_dim))
            .map(|i| 0.1 + (i as f32 % 10.0) * 0.01)
            .collect();
        let q_buffer = Rc::new(device.new_buffer_with_data(
            q_data.as_ptr() as *const std::ffi::c_void,
            q_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let k_data: Vec<half::f16> = (0..(n_kv_heads * max_seq * head_dim))
            .map(|i| half::f16::from_f32(0.01 * (i as f32 % 10.0)))
            .collect();
        let k_buffer = Rc::new(device.new_buffer_with_data(
            k_data.as_ptr() as *const std::ffi::c_void,
            kv_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let v_data: Vec<half::f16> = (0..(n_kv_heads * max_seq * head_dim))
            .map(|i| half::f16::from_f32(0.02 * (i as f32 % 10.0)))
            .collect();
        let v_buffer = Rc::new(device.new_buffer_with_data(
            v_data.as_ptr() as *const std::ffi::c_void,
            kv_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let out_size = n_heads * head_dim * std::mem::size_of::<f32>();
        let out_buffer = Rc::new(device.new_buffer(
            out_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        println!(
            "{:10} {:12} {:12} {:12} {:12}",
            "seq_k", "time (us)", "GB/s", "FLOPS", "Efficiency"
        );
        println!("{}", "-".repeat(60));

        let n_iterations = 100;

        for &seq_k in &seq_lengths {
            let mut total_time = std::time::Duration::ZERO;

            for _ in 0..n_iterations {
                let start = std::time::Instant::now();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();

                ctx.flash_attn_f16_encode(
                    encoder,
                    1,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    seq_k,
                    max_seq,
                    &q_buffer,
                    &k_buffer,
                    &v_buffer,
                    &out_buffer,
                );

                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();

                total_time += start.elapsed();
            }

            let avg_ns = total_time.as_nanos() as f64 / n_iterations as f64;
            let avg_s = avg_ns / 1e9;
            let avg_us = avg_ns / 1000.0;

            // Memory bandwidth calculation
            // Q: n_heads * head_dim * 4 bytes (FP32)
            // K: n_kv_heads * seq_k * head_dim * 2 bytes (F16)
            // V: n_kv_heads * seq_k * head_dim * 2 bytes (F16)
            // Output: n_heads * head_dim * 4 bytes (FP32)
            let q_bytes = n_heads * head_dim * 4;
            let k_bytes = n_kv_heads * seq_k * head_dim * 2;
            let v_bytes = n_kv_heads * seq_k * head_dim * 2;
            let out_bytes = n_heads * head_dim * 4;
            let total_bytes = (q_bytes + k_bytes + v_bytes + out_bytes) as f64;
            let gbps = total_bytes / avg_s / 1e9; // GB/s

            // FLOPS calculation
            // Q*K^T: n_heads * seq_k * (2 * head_dim) FLOPs (multiply + add)
            // Softmax: n_heads * seq_k * (5 + exp) FLOPs
            // V accumulation: n_heads * seq_k * (2 * head_dim) FLOPs
            // Total: approximately n_heads * seq_k * (4 * head_dim + 5)
            let flops =
                n_heads as f64 * seq_k as f64 * (4.0 * head_dim as f64 + 5.0);
            let gflops = flops / avg_s / 1e9; // GFLOPS

            // Theoretical peak for Apple Silicon GPU
            // M1 Pro/Max: ~5.2 TFLOPS (FP32), ~10.4 TFLOPS (FP16)
            // Memory bandwidth: ~200-400 GB/s (M1 Pro/Max)
            // Assuming 400 GB/s as peak bandwidth
            let peak_bandwidth = 400.0; // GB/s
            let bandwidth_eff = (gbps / peak_bandwidth) * 100.0;

            println!(
                "{:10} {:12.2} {:12.2} {:12.2} {:11.1}%",
                seq_k, avg_us, gbps, gflops, bandwidth_eff
            );
        }

        println!();
        println!("Memory access pattern:");
        println!("  Q:  {} KB (read once)", q_size / 1024);
        println!(
            "  K:  {} KB @ seq_k=1024",
            n_kv_heads * 1024 * head_dim * 2 / 1024
        );
        println!(
            "  V:  {} KB @ seq_k=1024",
            n_kv_heads * 1024 * head_dim * 2 / 1024
        );
        println!(
            "  Total: {} KB",
            (q_size + 2 * n_kv_heads * 1024 * head_dim * 2) / 1024
        );

        println!();
        println!(
            "Note: Assuming peak bandwidth of 400 GB/s for Apple Silicon GPU."
        );
        println!("Actual peak may vary based on device (M1/M2/M3 series).");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Metal not available");
    }
}
