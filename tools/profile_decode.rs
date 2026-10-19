// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Profile individual operations during decode

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

        println!("=== GPU Decode Profile ===");

        let result = std::thread::spawn(move || {
            let gpu_ctx = MetalContext::new().unwrap();
            let device = gpu_ctx.device();
            let ctx = InferenceContext::from_gguf_with_gpu_ctx(
                &model_path,
                gpu_ctx.clone(),
            )
            .unwrap();
            let config = ctx.model.config.clone();
            let n_layers = config.n_layers;
            let hidden_size = config.hidden_size;

            println!("Model: {} layers, hidden_size={}", n_layers, hidden_size);

            let mut kv = KVCacheGPU::new(device, &config, 2048);

            // Prefill
            let prefill_tokens: Vec<u32> = (1..=10).collect();
            let _ = ctx.model.forward_gpu_native_batched(
                &prefill_tokens,
                &mut kv,
                &gpu_ctx,
            );

            // Decode 3 tokens with detailed timing
            for token_num in 0..3 {
                println!("\n=== Token {} ===", token_num + 1);
                let token = (11 + token_num) as u32;

                // 1. Embedding lookup
                let t0 = Instant::now();
                let hidden = ctx.model.embedding_lookup_impl(token).unwrap();
                println!("  Embedding: {:?}", t0.elapsed());

                // Upload to GPU
                let t1 = Instant::now();
                let hidden_buffer =
                    std::rc::Rc::new(device.new_buffer_with_data(
                        hidden.as_ptr() as *const std::ffi::c_void,
                        (hidden_size * 4) as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    ));
                println!("  Upload: {:?}", t1.elapsed());

                // Process layers
                let mut layer_input = hidden_buffer.clone();

                // Measure total layer time
                let t_all_layers = Instant::now();

                for layer_idx in 0..n_layers {
                    let t_layer_start = Instant::now();

                    let queue = gpu_ctx.queue();
                    let command_buffer = queue.new_command_buffer();
                    let encoder = command_buffer.new_compute_command_encoder();

                    let t_encode = Instant::now();

                    // Just measure encode time
                    // Note: Actual forward is in forward_gpu_native_batched

                    let encode_time = t_encode.elapsed();

                    encoder.end_encoding();

                    let t_commit = Instant::now();
                    command_buffer.commit();
                    command_buffer.wait_until_completed();
                    let commit_time = t_commit.elapsed();

                    let total_time = t_layer_start.elapsed();

                    if layer_idx < 3 || layer_idx >= n_layers - 2 {
                        println!(
                            "    Layer {}: total={:?} encode={:?} commit={:?}",
                            layer_idx, total_time, encode_time, commit_time
                        );
                    }
                }

                println!(
                    "  All layers encode+commit: {:?}",
                    t_all_layers.elapsed()
                );

                kv.advance();
            }
        })
        .join()
        .unwrap();

        println!("\nProfile complete");
    }
}
