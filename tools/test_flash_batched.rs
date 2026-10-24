// Test batched Flash Attention vs vector version
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

        let n_heads = 32;
        let n_kv_heads = 8;
        let head_dim = 128;
        let max_seq = 2048;

        println!("Flash Attention Decode Kernel Comparison");
        println!(
            "Config: n_heads={}, n_kv_heads={}, head_dim={}",
            n_heads, n_kv_heads, head_dim
        );
        println!();

        // Create buffers
        let q_size = n_heads * head_dim * std::mem::size_of::<f32>();
        let kv_size =
            n_kv_heads * max_seq * head_dim * std::mem::size_of::<half::f16>();

        // Q: [n_heads, head_dim] - FP32 for easier verification
        let q_data: Vec<f32> = (0..(n_heads * head_dim))
            .map(|i| 0.1 + (i as f32 % 10.0) * 0.01)
            .collect();
        let q_buffer = Rc::new(device.new_buffer_with_data(
            q_data.as_ptr() as *const std::ffi::c_void,
            q_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // K: [n_kv_heads, max_seq, head_dim] - F16
        let k_data: Vec<half::f16> = (0..(n_kv_heads * max_seq * head_dim))
            .map(|i| half::f16::from_f32(0.01 * (i as f32 % 10.0)))
            .collect();
        let k_buffer = Rc::new(device.new_buffer_with_data(
            k_data.as_ptr() as *const std::ffi::c_void,
            kv_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // V: [n_kv_heads, max_seq, head_dim] - F16
        let v_data: Vec<half::f16> = (0..(n_kv_heads * max_seq * head_dim))
            .map(|i| half::f16::from_f32(0.02 * (i as f32 % 10.0)))
            .collect();
        let v_buffer = Rc::new(device.new_buffer_with_data(
            v_data.as_ptr() as *const std::ffi::c_void,
            kv_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Output buffers (one for each version)
        let out_size = n_heads * head_dim * std::mem::size_of::<f32>();
        let out_vec = Rc::new(device.new_buffer(
            out_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));
        let out_batched = Rc::new(device.new_buffer(
            out_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Test different sequence lengths
        let seq_lengths = vec![64, 128, 256, 512, 1024, 2048];
        let n_iterations = 50;

        println!(
            "{:10} {:15} {:15} {:12} {:12}",
            "seq_k", "vector (us)", "batched (us)", "speedup", "correct"
        );

        for &seq_k in &seq_lengths {
            // Test vector version
            let mut vec_time = std::time::Duration::ZERO;
            for _ in 0..n_iterations {
                let start = std::time::Instant::now();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();

                let success = ctx.flash_attn_f16_encode(
                    encoder, 1, n_heads, n_kv_heads, head_dim, seq_k, max_seq,
                    &q_buffer, &k_buffer, &v_buffer, &out_vec,
                );

                if !success {
                    eprintln!("Vector kernel failed for seq_k={}", seq_k);
                    continue;
                }

                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();

                vec_time += start.elapsed();
            }

            // Test batched version
            let mut batched_time = std::time::Duration::ZERO;
            for _ in 0..n_iterations {
                let start = std::time::Instant::now();
                let cb = queue.new_command_buffer();
                let encoder = cb.new_compute_command_encoder();

                let success = ctx.flash_attn_mma_batched_encode(
                    encoder,
                    n_heads,
                    n_kv_heads,
                    seq_k,
                    max_seq,
                    &q_buffer,
                    &k_buffer,
                    &v_buffer,
                    &out_batched,
                );

                if !success {
                    eprintln!("Batched kernel failed for seq_k={}", seq_k);
                    continue;
                }

                encoder.end_encoding();
                cb.commit();
                cb.wait_until_completed();

                batched_time += start.elapsed();
            }

            let vec_us = vec_time.as_micros() as f64 / n_iterations as f64;
            let batched_us =
                batched_time.as_micros() as f64 / n_iterations as f64;
            let speedup = vec_us / batched_us;

            // Check correctness - compare first few values
            let correct = unsafe {
                let vec_ptr = out_vec.contents() as *const f32;
                let batched_ptr = out_batched.contents() as *const f32;

                let mut max_diff = 0.0f32;
                for i in 0..10.min(n_heads * head_dim) {
                    let diff = (*vec_ptr.add(i) - *batched_ptr.add(i)).abs();
                    max_diff = max_diff.max(diff);
                }
                max_diff < 1e-3 // Allow small floating point differences
            };

            println!(
                "{:10} {:15.2} {:15.2} {:12.2}x {:12}",
                seq_k,
                vec_us,
                batched_us,
                speedup,
                if correct { "✓" } else { "✗" }
            );
        }

        println!();
        println!("Memory bandwidth analysis:");
        println!("  Q:  {} KB", q_size / 1024);
        println!("  K:  {} KB (seq_k=1024)", kv_size / 1024);
        println!("  V:  {} KB (seq_k=1024)", kv_size / 1024);
        println!("  Total per decode: {} KB", (q_size + 2 * kv_size) / 1024);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Metal not available");
    }
}
