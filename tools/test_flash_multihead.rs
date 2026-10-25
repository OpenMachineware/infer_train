// Compare Flash Attention kernel variants
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
        let seq_k = 1024;
        let max_seq = 2048;

        let q_size = n_heads * head_dim * 4;
        let kv_size = n_kv_heads * max_seq * head_dim * 2;

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

        let out_size = n_heads * head_dim * 4;
        let out_vec = Rc::new(device.new_buffer(
            out_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));
        let out_multi = Rc::new(device.new_buffer(
            out_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Test standard kernel
        let n_iterations = 100;
        let mut vec_time = std::time::Duration::ZERO;

        for _ in 0..n_iterations {
            let start = std::time::Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();

            ctx.flash_attn_f16_encode(
                encoder, 1, n_heads, n_kv_heads, head_dim, seq_k, max_seq,
                &q_buffer, &k_buffer, &v_buffer, &out_vec,
            );

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            vec_time += start.elapsed();
        }

        // Test multihead kernel
        let mut multi_time = std::time::Duration::ZERO;
        for _ in 0..n_iterations {
            let start = std::time::Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();

            let success = ctx.flash_attn_multihead_encode(
                encoder, 1, n_heads, n_kv_heads, head_dim, seq_k, max_seq,
                &q_buffer, &k_buffer, &v_buffer, &out_multi,
            );

            if !success {
                eprintln!("Multihead kernel not available");
                return;
            }

            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            multi_time += start.elapsed();
        }

        let vec_us = vec_time.as_micros() as f64 / n_iterations as f64;
        let multi_us = multi_time.as_micros() as f64 / n_iterations as f64;
        let speedup = vec_us / multi_us;

        // Calculate bandwidth
        let q_bytes = n_heads * head_dim * 4;
        let k_bytes = n_kv_heads * seq_k * head_dim * 2;
        let v_bytes = n_kv_heads * seq_k * head_dim * 2;
        let out_bytes = n_heads * head_dim * 4;
        let total_bytes = (q_bytes + k_bytes + v_bytes + out_bytes) as f64;

        let vec_gbps = total_bytes / (vec_us * 1e-6);
        let multi_gbps = total_bytes / (multi_us * 1e-6);

        println!("Flash Attention Kernel Comparison (seq_k={})", seq_k);
        println!();
        println!("Standard kernel:");
        println!("  Time: {:.2} us", vec_us);
        println!(
            "  Bandwidth: {:.2} GB/s ({:.1}% efficiency)",
            vec_gbps,
            vec_gbps / 400.0 * 100.0
        );
        println!();
        println!("Multihead kernel:");
        println!("  Time: {:.2} us", multi_us);
        println!(
            "  Bandwidth: {:.2} GB/s ({:.1}% efficiency)",
            multi_gbps,
            multi_gbps / 400.0 * 100.0
        );
        println!();
        println!("Speedup: {:.2}x", speedup);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Metal not available");
    }
}
