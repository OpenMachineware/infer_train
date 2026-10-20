// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Profile Flash Attention performance

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::rc::Rc;

        println!("=== Flash Attention Profile ===\n");

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let queue = gpu_ctx.queue();

        // Test dimensions from DeepSeek-R1-Distill-Qwen-1.5B
        let n_heads = 12u64;
        let n_kv_heads = 2u64; // MQA/GQA
        let head_dim = 128u64;
        let kv_seq_len = 128u64; // Context length

        println!("Config:");
        println!("  n_heads: {}", n_heads);
        println!("  n_kv_heads: {}", n_kv_heads);
        println!("  head_dim: {}", head_dim);
        println!("  kv_seq_len: {}", kv_seq_len);

        // Create test buffers
        let q_size = n_heads * head_dim * 2; // F16
        let k_size = n_kv_heads * kv_seq_len * head_dim * 2;
        let v_size = n_kv_heads * kv_seq_len * head_dim * 2;
        let out_size = n_heads * head_dim * 4; // F32

        let q_buffer =
            Rc::new(device.new_buffer(
                q_size,
                metal::MTLResourceOptions::StorageModeShared,
            ));
        let k_buffer =
            Rc::new(device.new_buffer(
                k_size,
                metal::MTLResourceOptions::StorageModeShared,
            ));
        let v_buffer =
            Rc::new(device.new_buffer(
                v_size,
                metal::MTLResourceOptions::StorageModeShared,
            ));
        let out_buffer = Rc::new(device.new_buffer(
            out_size,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Warmup
        println!("\nWarming up...");
        for _ in 0..10 {
            let cmd = queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();
            gpu_ctx.flash_attn_f16_encode(
                encoder,
                1, // n_seqs
                n_heads as usize,
                n_kv_heads as usize,
                head_dim as usize,
                1, // seq_len = 1 for decode
                kv_seq_len as usize,
                &q_buffer,
                &k_buffer,
                &v_buffer,
                &out_buffer,
            );
            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // Benchmark
        println!("\n=== Flash Attention Decode (seq_len=1) ===");
        let mut times = Vec::new();
        for i in 0..100 {
            let cmd = queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();
            gpu_ctx.flash_attn_f16_encode(
                encoder,
                1,
                n_heads as usize,
                n_kv_heads as usize,
                head_dim as usize,
                1,
                kv_seq_len as usize,
                &q_buffer,
                &k_buffer,
                &v_buffer,
                &out_buffer,
            );
            encoder.end_encoding();

            let start = std::time::Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            times.push(start.elapsed());

            if i < 5 || i >= 95 {
                println!("  Run {}: {:?}", i + 1, times.last().unwrap());
            }
        }

        let avg: std::time::Duration =
            times.iter().sum::<std::time::Duration>() / times.len() as u32;
        println!("\n  Average: {:?}", avg);

        // Calculate bandwidth
        // K read: n_kv_heads * kv_seq_len * head_dim * 2 bytes
        // V read: n_kv_heads * kv_seq_len * head_dim * 2 bytes
        // Total read: 2 * 2 * 2 * 128 * 128 = 131072 bytes = 128 KB
        let k_bytes = n_kv_heads * kv_seq_len * head_dim * 2;
        let v_bytes = n_kv_heads * kv_seq_len * head_dim * 2;
        let total_bytes = k_bytes + v_bytes;
        let bandwidth_gbps = total_bytes as f64 / avg.as_secs_f64() / 1e9;

        println!("\n  KV cache read: {} KB", total_bytes / 1024);
        println!("  Bandwidth: {:.1} GB/s", bandwidth_gbps);

        // Test with longer context
        println!("\n=== Testing with longer context ===");
        for kv_len in [256, 512, 1024, 2048].iter() {
            let k_buf = Rc::new(device.new_buffer(
                n_kv_heads * kv_len * head_dim * 2,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            let v_buf = Rc::new(device.new_buffer(
                n_kv_heads * kv_len * head_dim * 2,
                metal::MTLResourceOptions::StorageModeShared,
            ));

            // Warmup
            let cmd = queue.new_command_buffer();
            let encoder = cmd.new_compute_command_encoder();
            gpu_ctx.flash_attn_f16_encode(
                encoder,
                1,
                n_heads as usize,
                n_kv_heads as usize,
                head_dim as usize,
                1,
                *kv_len as usize,
                &q_buffer,
                &k_buf,
                &v_buf,
                &out_buffer,
            );
            encoder.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Benchmark
            let mut times = Vec::new();
            for _ in 0..50 {
                let cmd = queue.new_command_buffer();
                let encoder = cmd.new_compute_command_encoder();
                gpu_ctx.flash_attn_f16_encode(
                    encoder,
                    1,
                    n_heads as usize,
                    n_kv_heads as usize,
                    head_dim as usize,
                    1,
                    *kv_len as usize,
                    &q_buffer,
                    &k_buf,
                    &v_buf,
                    &out_buffer,
                );
                encoder.end_encoding();
                let start = std::time::Instant::now();
                cmd.commit();
                cmd.wait_until_completed();
                times.push(start.elapsed());
            }

            let avg =
                times.iter().sum::<std::time::Duration>() / times.len() as u32;
            let total_bytes = 2 * n_kv_heads * kv_len * head_dim * 2;
            let bw = total_bytes as f64 / avg.as_secs_f64() / 1e9;

            println!("  kv_seq_len={}: {:?} ({:.1} GB/s)", kv_len, avg, bw);
        }

        println!("\n=== Summary ===");
        println!("Flash Attention decode latency at kv_seq_len=128: {:?}", avg);
        println!("Per-layer contribution (28 layers): {:?}", avg * 28);
        println!("Compare with total decode time: ~27ms");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        eprintln!("Requires Metal support on aarch64");
    }
}
