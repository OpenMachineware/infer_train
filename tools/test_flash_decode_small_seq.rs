// Test flash decode GPU kernel for small sequence lengths
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        // Create Metal context
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        let n_heads = 16;
        let n_kv_heads = 8;
        let head_dim = 128;
        let max_seq = 100;

        // Create buffers
        let q_size = n_heads * head_dim * std::mem::size_of::<f32>();
        let kv_size = n_kv_heads * max_seq * head_dim * std::mem::size_of::<half::f16>();

        let q_buffer = std::rc::Rc::new(ctx.device().new_buffer_with_data(
            vec![0.1f32; n_heads * head_dim].as_ptr() as *const std::ffi::c_void,
            q_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Create KV cache with some values
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

        // Test different sequence lengths
        for seq_len in [1, 2, 3, 4, 5, 10, 20, 50] {
            println!("\n=== seq_len = {} ===", seq_len);

            let output_buffer = ctx
                .flash_attn_f16_gpu(
                    1,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    seq_len,
                    max_seq,
                    &q_buffer,
                    &k_buffer,
                    &v_buffer,
                )
                .expect("Flash attention failed");

            // Read output
            let output_ptr = output_buffer.contents() as *const f32;
            let output_len = n_heads * head_dim;
            let output: Vec<f32> =
                unsafe { std::slice::from_raw_parts(output_ptr, output_len).to_vec() };

            let num_nan = output.iter().filter(|&&v| v.is_nan()).count();
            let num_inf = output.iter().filter(|&&v| v.is_infinite()).count();
            let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
            let mean = output.iter().sum::<f32>() / output.len() as f32;

            println!(
                "  output: len={}, nan={}, inf={}, range=[{:.4}, {:.4}], mean={:.4}",
                output.len(),
                num_nan,
                num_inf,
                min,
                max,
                mean
            );

            if num_nan > 0 {
                println!("  WARNING: NaN detected!");
                // Find first NaN
                for (i, &v) in output.iter().enumerate() {
                    if v.is_nan() {
                        println!("    First NaN at index {}", i);
                        break;
                    }
                }
            }
        }

        println!("\n=== Test complete ===");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Metal not available");
    }
}
