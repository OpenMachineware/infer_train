// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile Flash Attention execution time

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = args[1].clone();

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;
        use std::time::Instant;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            &model_path,
            gpu_ctx.clone(),
        )
        .unwrap();
        let config = ctx.model.config.clone();
        let mut kv = KVCacheGPU::new(&device, &config, 2048);

        // Prefill to position 10
        let prefill_tokens: Vec<u32> = (1..=10).collect();
        let _ = ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv,
            &gpu_ctx,
        );

        let hidden_size = config.hidden_size;
        let head_dim = config.head_dim;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;

        println!("=== Flash Attention Performance ===");
        println!("Hidden size: {}, Head dim: {}", hidden_size, head_dim);
        println!("N heads: {}, N KV heads: {}", n_heads, n_kv_heads);
        println!("KV cache position: {}", kv.seq_len());

        // Create test Q, K, V buffers
        let q_size = n_heads * head_dim;
        let k_size = n_kv_heads * head_dim;
        let v_size = n_kv_heads * head_dim;

        let q_data: Vec<f32> = vec![0.1; q_size];
        let q_buffer = std::rc::Rc::new(device.new_buffer_with_data(
            q_data.as_ptr() as *const std::ffi::c_void,
            (q_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let k_buffer = std::rc::Rc::new(device.new_buffer_with_data(
            q_data.as_ptr() as *const std::ffi::c_void,
            (k_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let v_buffer = std::rc::Rc::new(device.new_buffer_with_data(
            q_data.as_ptr() as *const std::ffi::c_void,
            (v_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let output_buffer = std::rc::Rc::new(device.new_buffer(
            (q_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let queue = gpu_ctx.queue();
        let mut times = Vec::new();

        // Measure
        for _ in 0..20 {
            let start = Instant::now();
            let command_buffer = queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Note: This won't work without proper KV cache setup
            // Just showing the concept
            gpu_ctx.flash_decode_encode(
                encoder,
                &q_buffer,
                &k_buffer,
                &v_buffer,
                &output_buffer,
                n_heads,
                n_kv_heads,
                head_dim,
                1,            // seq_len
                kv.seq_len(), // kv_seq_len
                0,            // kv_offset
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            times.push(start.elapsed());
        }

        let avg: std::time::Duration =
            times.iter().sum::<std::time::Duration>() / times.len() as u32;
        println!("  Avg time: {:?}", avg);
        println!("  For 28 layers: {:?}", avg * 28);
    }
}
