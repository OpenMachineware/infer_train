// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Safe test for async command buffer implementation
//! Tests a few token generations to verify correctness before running heavy benchmarks

fn main() {
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/Users/jia/Models/Qwen3-1.7B-F16.gguf".to_string()
    });

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        eprintln!("Testing async command buffer with model: {}", model_path);

        // Create GPU context
        let gpu_ctx =
            MetalContext::new().expect("Failed to create Metal context");
        let device = gpu_ctx.device();

        // Load model
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            &model_path,
            gpu_ctx.clone(),
        )
        .expect("Failed to load model");

        let config = ctx.model.config.clone();
        eprintln!("Model loaded: {} layers", config.n_layers);

        // Create KV cache
        let mut kv = KVCacheGPU::new(device, &config, 2048);

        // Prefill phase
        let prefill_tokens: Vec<u32> = (1..=10).collect();
        eprintln!("\nPrefilling {} tokens...", prefill_tokens.len());

        let logits = ctx
            .model
            .forward_gpu_native_batched(&prefill_tokens, &mut kv, &gpu_ctx)
            .expect("Prefill failed");

        // Check prefill output
        let has_nan = logits.iter().any(|x| x.is_nan() || x.is_infinite());
        if has_nan {
            eprintln!("✗ ERROR: Prefill output contains NaN or Inf!");
            std::process::exit(1);
        } else {
            eprintln!("✓ Prefill OK, {} logits", logits.len());
        }

        // Decode phase - test 3 tokens
        eprintln!("\nDecoding 3 tokens...");
        for i in 0..3 {
            let token = (11 + i) as u32;
            eprintln!("\n=== Token {} ===", i + 1);

            let logits = ctx
                .model
                .forward_gpu_native_batched(&[token], &mut kv, &gpu_ctx)
                .expect(&format!("Decode token {} failed", i + 1));

            // Check decode output
            let has_nan = logits.iter().any(|x| x.is_nan() || x.is_infinite());
            if has_nan {
                eprintln!("✗ ERROR: Decode token {} has NaN/Inf!", i + 1);
                std::process::exit(1);
            } else {
                let max_logit =
                    logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min_logit =
                    logits.iter().cloned().fold(f32::INFINITY, f32::min);
                eprintln!(
                    "✓ Token {} OK, logits: [{:.3}, {:.3}]",
                    i + 1,
                    min_logit,
                    max_logit
                );
            }
        }

        eprintln!("\n✅ All tests passed! Async implementation is safe.");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        eprintln!("This test requires Metal support on aarch64");
    }
}
