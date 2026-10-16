// Benchmark flash decode kernel performance
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        let n_heads = 16;
        let n_kv_heads = 8;
        let head_dim = 128;
        let max_seq = 2048;

        // Create buffers
        let q_size = n_heads * head_dim * std::mem::size_of::<f32>();
        let kv_size = n_kv_heads * max_seq * head_dim * std::mem::size_of::<half::f16>();

        let q_buffer = std::rc::Rc::new(ctx.device().new_buffer_with_data(
            vec![0.1f32; n_heads * head_dim].as_ptr() as *const std::ffi::c_void,
            q_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let k_data: Vec<half::f16> = (0..(n_kv_heads * max_seq * head_dim))
            .map(|i| half::f16::from_f32(0.01 * (i as f32 % 10.0)))
            .collect();
        let k_buffer = std::rc::Rc::new(ctx.device().new_buffer_with_data(
            k_data.as_ptr() as *const std::ffi::c_void,
            kv_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let v_data: Vec<half::f16> = (0..(n_kv_heads * max_seq * head_dim))
            .map(|i| half::f16::from_f32(0.02 * (i as f32 % 10.0)))
            .collect();
        let v_buffer = std::rc::Rc::new(ctx.device().new_buffer_with_data(
            v_data.as_ptr() as *const std::ffi::c_void,
            kv_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Benchmark different sequence lengths
        println!("Flash Decode Kernel Performance (n_heads={}, head_dim={})", n_heads, head_dim);
        println!("{:10} {:12} {:12} {:12}", "seq_k", "time (us)", "GFLOPS", "GB/s");

        let n_iterations = 100;

        for &seq_k in &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
            let mut total_time = std::time::Duration::ZERO;

            for _ in 0..n_iterations {
                let start = std::time::Instant::now();
                let _ = ctx
                    .flash_attn_f16_gpu(
                        1,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        seq_k,
                        max_seq,
                        &q_buffer,
                        &k_buffer,
                        &v_buffer,
                    )
                    .expect("Flash attention failed");
                total_time += start.elapsed();
            }

            let avg_time = total_time.as_nanos() as f64 / n_iterations as f64;
            let avg_us = avg_time / 1000.0;

            // Compute FLOPS: 2 * head_dim * seq_k * n_heads (Q*K^T + softmax + V accumulation)
            let flops = 2.0 * head_dim as f64 * seq_k as f64 * n_heads as f64 * 2.0; // x2 for QK and V
            let gflops = flops / avg_time;

            // Memory bandwidth: Q (n_heads * head_dim * 4) + K (n_kv_heads * seq_k * head_dim * 2) + V (same)
            let q_bytes = n_heads * head_dim * 4;
            let kv_bytes = n_kv_heads * seq_k * head_dim * 2;
            let total_bytes = (q_bytes + 2 * kv_bytes) as f64;
            let gbps = total_bytes / avg_time;

            println!("{:10} {:12.2} {:12.2} {:12.2}", seq_k, avg_us, gflops, gbps);
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Metal not available");
    }
}
