// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Detailed NaN investigation for decode

use std::time::Instant;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext};

        // Load model using InferenceContext (handles Metal initialization properly)
        let model_path = std::env::var("MODEL_PATH")
            .unwrap_or_else(|_| "models/Qwen3-0.6B-Q4_K_M.gguf".to_string());
        println!("Loading model from: {}", model_path);

        let mut ctx = InferenceContext::from_gguf(&model_path, Device::Gpu)
            .expect("Failed to load model on GPU");

        let vocab_size = ctx.config.vocab_size;
        let hidden_size = ctx.config.hidden_size;
        let n_layers = ctx.layer_count();
        let n_heads = ctx.config.n_heads;
        let n_kv_heads = ctx.config.n_kv_heads;
        let head_dim = ctx.config.head_dim;

        println!(
            "Model: vocab={}, hidden={}, layers={}, heads={}/{}, head_dim={}",
            vocab_size, hidden_size, n_layers, n_heads, n_kv_heads, head_dim
        );

        // Test input
        let input_tokens: Vec<u32> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        ];

        // Run multiple times to catch non-deterministic NaN
        for run in 1..=20 {
            // Prefill
            let prefill_start = std::time::Instant::now();
            let logits = match ctx.forward_prefill_batch(&input_tokens) {
                Some(l) => l,
                None => {
                    println!("Run {}: Prefill failed", run);
                    continue;
                }
            };
            let prefill_time = prefill_start.elapsed();

            let prefill_nan = logits.iter().filter(|&&x| x.is_nan()).count();

            if prefill_nan > 0 {
                println!("Run {}: Prefill has {} NaN", run, prefill_nan);
            }

            // Decode - check each token
            let mut any_nan = false;
            for i in 0..10 {
                let token = (65 + i) as u32;
                let decode_start = std::time::Instant::now();
                let result = ctx.forward_gpu_batched(&[token]);
                let decode_time = decode_start.elapsed();

                if let Some(logits) = &result {
                    let nan_count =
                        logits.iter().filter(|&&x| x.is_nan()).count();

                    if nan_count > 0 {
                        any_nan = true;
                        println!(
                            "Run {}, Token {}: NaN: {}/{}, time: {:?}",
                            run,
                            i + 1,
                            nan_count,
                            logits.len(),
                            decode_time
                        );
                    }
                }
            }

            if !any_nan {
                println!("Run {}: OK (prefill: {:?})", run, prefill_time);
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Metal not available");
    }
}
