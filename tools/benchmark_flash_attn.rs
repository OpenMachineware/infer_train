// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test Flash Attention performance for decode

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        println!("=== Flash Attention Decode Performance ===\n");

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        // DeepSeek-R1-Distill-Qwen-1.5B config
        let n_heads = 12;
        let n_kv_heads = 2; // GQA
        let head_dim = 128;
        let seq_len = 1; // Decode

        println!("Config:");
        println!("  n_heads: {}", n_heads);
        println!("  n_kv_heads: {}", n_kv_heads);
        println!("  head_dim: {}", head_dim);
        println!("  seq_len: {}", seq_len);
        println!();

        // Create test buffers
        let q_size = n_heads * head_dim;
        let k_size = n_kv_heads * head_dim;
        let v_size = n_kv_heads * head_dim;

        let q_data: Vec<f32> = vec![0.1; q_size];
        let k_data: Vec<f32> = vec![0.1; k_size];
        let v_data: Vec<f32> = vec![0.1; v_size];

        let q_buffer = std::rc::Rc::new(device.new_buffer_with_data(
            q_data.as_ptr() as *const std::ffi::c_void,
            (q_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let k_buffer = std::rc::Rc::new(device.new_buffer_with_data(
            k_data.as_ptr() as *const std::ffi::c_void,
            (k_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let v_buffer = std::rc::Rc::new(device.new_buffer_with_data(
            v_data.as_ptr() as *const std::ffi::c_void,
            (v_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let output_buffer = std::rc::Rc::new(device.new_buffer(
            (q_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Warmup
        for _ in 0..5 {
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            gpu_ctx.flash_attn_f16_encode(
                encoder,
                1, // n_seqs
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                n_kv_heads * head_dim, // kv_stride
                &q_buffer,
                &k_buffer,
                &v_buffer,
                &output_buffer,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Measure
        let iterations = 50;
        let mut times = Vec::new();
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            gpu_ctx.flash_attn_f16_encode(
                encoder,
                1, // n_seqs
                n_heads,
                n_kv_heads,
                head_dim,
                seq_len,
                n_kv_heads * head_dim, // kv_stride
                &q_buffer,
                &k_buffer,
                &v_buffer,
                &output_buffer,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            times.push(start.elapsed());
        }

        let avg: std::time::Duration =
            times.iter().sum::<std::time::Duration>() / iterations as u32;

        println!("Flash Attention (single head, decode):");
        println!("  Avg time: {:?}", avg);
        println!();

        // Test with KV cache (simulating actual decode)
        let max_seq = 2048;
        let kv_cache_size = max_seq * n_kv_heads * head_dim;
        let k_cache = std::rc::Rc::new(device.new_buffer(
            (kv_cache_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));
        let v_cache = std::rc::Rc::new(device.new_buffer(
            (kv_cache_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Measure with cache
        let mut times_with_cache = Vec::new();
        for pos in 0..iterations {
            let start = std::time::Instant::now();
            let cb = queue.new_command_buffer();
            let encoder = cb.new_compute_command_encoder();
            gpu_ctx.flash_attn_f16_encode(
                encoder,
                1, // n_seqs
                n_heads,
                n_kv_heads,
                head_dim,
                pos + 1,               // seq_len = pos + 1
                n_kv_heads * head_dim, // kv_stride
                &q_buffer,
                &k_cache,
                &v_cache,
                &output_buffer,
            );
            encoder.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            times_with_cache.push(start.elapsed());
        }

        let avg_with_cache: std::time::Duration =
            times_with_cache.iter().sum::<std::time::Duration>()
                / iterations as u32;

        println!("Flash Attention (with KV cache, pos=0..50):");
        println!("  Avg time: {:?}", avg_with_cache);
        println!();

        // Estimate impact on decode
        println!("=== Impact on Decode ===\n");
        println!("Per layer Flash Attention time: {:?}", avg);
        println!("28 layers total: {:?}", avg * 28);
        println!();
        println!("Current decode time: 14.90ms");
        println!(
            "Flash Attention contribution: {:.1}%",
            (avg.as_secs_f64() * 1000.0 * 28.0 / 14.90) * 100.0
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support");
    }
}
