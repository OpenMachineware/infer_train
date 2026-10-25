// Debug batched Flash Attention correctness
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
        let max_seq = 64; // Small for debugging
        let seq_k = 16; // Small for debugging

        // Create simple test data
        let q_data: Vec<f32> = (0..(n_heads * head_dim))
            .map(|i| 0.1 + (i as f32 % 10.0) * 0.01)
            .collect();
        let q_buffer = Rc::new(device.new_buffer_with_data(
            q_data.as_ptr() as *const std::ffi::c_void,
            (n_heads * head_dim * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // K/V data
        let k_data: Vec<half::f16> = (0..(n_kv_heads * max_seq * head_dim))
            .map(|i| half::f16::from_f32(0.01 * (i as f32 % 10.0)))
            .collect();
        let k_buffer = Rc::new(device.new_buffer_with_data(
            k_data.as_ptr() as *const std::ffi::c_void,
            (n_kv_heads * max_seq * head_dim * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let v_data: Vec<half::f16> = (0..(n_kv_heads * max_seq * head_dim))
            .map(|i| half::f16::from_f32(0.02 * (i as f32 % 10.0)))
            .collect();
        let v_buffer = Rc::new(device.new_buffer_with_data(
            v_data.as_ptr() as *const std::ffi::c_void,
            (n_kv_heads * max_seq * head_dim * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Output buffers
        let out_size = n_heads * head_dim * 4;
        let out_vec = Rc::new(device.new_buffer(
            out_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));
        let out_batched = Rc::new(device.new_buffer(
            out_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Run vector version
        {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            ctx.flash_attn_f16_encode(
                encoder, 1, n_heads, n_kv_heads, head_dim, seq_k, max_seq,
                &q_buffer, &k_buffer, &v_buffer, &out_vec,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Run batched version
        {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            ctx.flash_attn_mma_batched_encode(
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
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Compare outputs
        unsafe {
            let vec_ptr = out_vec.contents() as *const f32;
            let batched_ptr = out_batched.contents() as *const f32;

            println!("Comparing first 16 values of head 0:");
            println!(
                "{:10} {:15} {:15} {:12}",
                "idx", "vector", "batched", "diff"
            );
            for i in 0..16 {
                let v = *vec_ptr.add(i);
                let b = *batched_ptr.add(i);
                let diff = (v - b).abs();
                println!("{:10} {:15.6} {:15.6} {:12.6}", i, v, b, diff);
            }

            println!("\nComparing head 4 (different KV head mapping):");
            let head4_offset = 4 * head_dim;
            println!(
                "{:10} {:15} {:15} {:12}",
                "idx", "vector", "batched", "diff"
            );
            for i in 0..16 {
                let v = *vec_ptr.add(head4_offset + i);
                let b = *batched_ptr.add(head4_offset + i);
                let diff = (v - b).abs();
                println!("{:10} {:15.6} {:15.6} {:12.6}", i, v, b, diff);
            }

            // Check for NaN
            let mut has_nan_vec = false;
            let mut has_nan_batched = false;
            for i in 0..(n_heads * head_dim) {
                let v = *vec_ptr.add(i);
                let b = *batched_ptr.add(i);
                if v.is_nan() || v.is_infinite() {
                    has_nan_vec = true;
                    if i < 100 {
                        println!("Vector NaN/Inf at index {}: {}", i, v);
                    }
                }
                if b.is_nan() || b.is_infinite() {
                    has_nan_batched = true;
                    if i < 100 {
                        println!("Batched NaN/Inf at index {}: {}", i, b);
                    }
                }
            }
            println!("\nVector has NaN/Inf: {}", has_nan_vec);
            println!("Batched has NaN/Inf: {}", has_nan_batched);

            // Overall comparison
            let mut max_diff = 0.0f32;
            let mut avg_diff = 0.0f32;
            for i in 0..(n_heads * head_dim) {
                let v = *vec_ptr.add(i);
                let b = *batched_ptr.add(i);
                let diff = (v - b).abs();
                max_diff = max_diff.max(diff);
                avg_diff += diff;
            }
            avg_diff /= (n_heads * head_dim) as f32;
            println!("\nMax diff: {:.6}", max_diff);
            println!("Avg diff: {:.6}", avg_diff);
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Metal not available");
    }
}
