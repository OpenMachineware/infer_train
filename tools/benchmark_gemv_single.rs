// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Benchmark single Q5_K GEMV kernel performance
// Compare with llama.cpp's kernel

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::model::tensor::DataType;
        use infer_train::quant::types::BlockQ5K;
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use metal::MTLSize;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        // Test sizes from Qwen3-0.6B
        let m = 1024; // hidden_size
        let k = 1024; // hidden_size
        let nb = k / 256;

        // Create test data
        let n_blocks = m * nb;
        let weights: Vec<BlockQ5K> = (0..n_blocks)
            .map(|i| BlockQ5K {
                d: half::f16::from_f32(0.5).to_bits(),
                dmin: 0u16,
                scales: [0u8; 12],
                qh: [((i % 256) as u8); 32],
                qs: [0u8; 128],
            })
            .collect();

        let input: Vec<f32> =
            (0..k).map(|i| (i as f32 % 100.0) / 100.0).collect();

        // Create buffers
        let weights_buffer = device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (n_blocks * std::mem::size_of::<BlockQ5K>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (k * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Warm up
        for _ in 0..10 {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            gpu_ctx.mv_encode(
                &encoder,
                &weights_buffer,
                &input_buffer,
                &output_buffer,
                m,
                k,
                DataType::Q5_K,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Benchmark
        let n_iter = 100;
        let start = std::time::Instant::now();
        for _ in 0..n_iter {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            gpu_ctx.mv_encode(
                &encoder,
                &weights_buffer,
                &input_buffer,
                &output_buffer,
                m,
                k,
                DataType::Q5_K,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let elapsed = start.elapsed();

        let time_per_iter = elapsed / n_iter;
        let flops = 2 * m * k; // multiply-add = 2 FLOPs
        let gflops = (flops as f64 / 1e9) / time_per_iter.as_secs_f64();

        println!("=== Q5_K GEMV Performance ===");
        println!("Size: {} x {}", m, k);
        println!("Time per iteration: {:?}", time_per_iter);
        println!("Throughput: {:.2} GFLOPS", gflops);

        // Calculate FLOPS per second
        let ops_per_sec = flops as f64 / time_per_iter.as_secs_f64();
        println!("Operations: {:.2} M FLOPs/sec", ops_per_sec / 1e6);

        // Compare with theoretical
        // M3: ~1000 GFLOPS for F16, Q5_K is more complex so ~500 GFLOPS?
        let theoretical_gflops = 500.0;
        let efficiency = (gflops / theoretical_gflops) * 100.0;
        println!("Efficiency: {:.1}% of theoretical", efficiency);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Requires aarch64+metal");
    }
}
