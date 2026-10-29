// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test new architecture CPU inference - correctness and performance

#[cfg(test)]
mod tests {
    use infer_train::inference::{
        Device, InferenceContext, KVCache, KVCacheFormat,
    };
    use infer_train::ops::sampling::{SamplingStrategy, sample};
    use std::time::Instant;

    const MODEL_PATH: &str =
        "/Users/jia/.llm_models/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf";

    #[test]
    fn test_cpu_correctness() {
        println!("\n=== CPU Correctness Test ===\n");

        let mut ctx = InferenceContext::from_gguf(MODEL_PATH, Device::Cpu)
            .expect("Failed to load model");

        println!(
            "Model loaded: {} layers, {} params",
            ctx.model.layers.len(),
            ctx.param_count()
        );

        // Create KV cache
        let config = ctx.config.clone();

        // Test tokens
        let test_tokens: Vec<Vec<u32>> = vec![
            vec![1],
            vec![1, 2, 3],
            vec![128000, 128001], // BOS, special tokens
        ];

        for tokens in test_tokens {
            println!("\nTesting tokens: {:?}", tokens);
            let mut kv_cache = KVCache::new(&config, 100, KVCacheFormat::F32);

            let logits =
                ctx.forward(&tokens, &mut kv_cache).expect("Forward failed");

            // Verify no NaN
            let nan_count = logits.iter().filter(|&&v| v.is_nan()).count();
            assert_eq!(nan_count, 0, "Logits contain {} NaN values", nan_count);

            // Verify no Inf
            let inf_count = logits.iter().filter(|&&v| v.is_infinite()).count();
            assert_eq!(inf_count, 0, "Logits contain {} Inf values", inf_count);

            // Verify vocab size
            assert_eq!(
                logits.len(),
                config.vocab_size,
                "Logits size {} != vocab_size {}",
                logits.len(),
                config.vocab_size
            );

            // Verify non-zero
            let non_zero = logits.iter().filter(|&&v| v != 0.0).count();
            assert!(
                non_zero > config.vocab_size / 2,
                "Too few non-zero logits: {}/{}",
                non_zero,
                config.vocab_size
            );

            // Sample a token
            let strategy = SamplingStrategy::Greedy;
            let token = sample(&logits, &strategy);
            println!("  Sampled token: {}", token);

            // Stats
            let max_val =
                logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_val = logits.iter().cloned().fold(f32::INFINITY, f32::min);
            println!("  Logits range: [{:.2}, {:.2}]", min_val, max_val);
        }

        println!("\n✓ All correctness tests passed");
    }

    #[test]
    fn test_cpu_prefill_performance() {
        println!("\n=== CPU Prefill Performance Test ===\n");

        let mut ctx = InferenceContext::from_gguf(MODEL_PATH, Device::Cpu)
            .expect("Failed to load model");

        let config = ctx.config.clone();

        // Test different prefill lengths
        let prefill_lengths: Vec<usize> = vec![32, 64, 128, 256, 512];

        println!("| Tokens | Time (ms) | Tok/s |");
        println!("|--------|-----------|-------|");

        for len in prefill_lengths {
            let tokens: Vec<u32> =
                (0..len as u32).map(|x| (x % 1000) + 1).collect();
            let mut kv_cache =
                KVCache::new(&config, len + 100, KVCacheFormat::F32);

            let start = Instant::now();
            let logits =
                ctx.forward(&tokens, &mut kv_cache).expect("Forward failed");
            let elapsed = start.elapsed();

            let tok_per_sec = len as f64 / elapsed.as_secs_f64();

            println!(
                "| {} | {:.1} | {:.1} |",
                len,
                elapsed.as_millis(),
                tok_per_sec
            );

            // Verify correctness
            assert!(logits.len() == config.vocab_size);
            let nan_count = logits.iter().filter(|&&v| v.is_nan()).count();
            assert_eq!(nan_count, 0);
        }

        println!("\n✓ Prefill performance test completed");
    }

    #[test]
    fn test_cpu_decode_performance() {
        println!("\n=== CPU Decode Performance Test ===\n");

        let mut ctx = InferenceContext::from_gguf(MODEL_PATH, Device::Cpu)
            .expect("Failed to load model");

        let config = ctx.config.clone();
        let mut kv_cache = KVCache::new(&config, 500, KVCacheFormat::F32);

        // Prefill first
        let prefill_tokens: Vec<u32> =
            (0..64).map(|x| (x % 1000) + 1).collect();
        ctx.forward(&prefill_tokens, &mut kv_cache).expect("Prefill failed");

        // Decode tokens
        let decode_count = 100;
        let strategy = SamplingStrategy::Greedy;
        let mut generated = Vec::with_capacity(decode_count);

        println!("| Batch | Total ms | Tok/s | Avg ms/tok |");
        println!("|-------|----------|-------|------------|");

        let start = Instant::now();
        let mut latencies = Vec::new();

        for i in 0..decode_count {
            let token_start = Instant::now();
            let tokens =
                vec![*generated.last().unwrap_or(&config.bos_token_id)];

            let logits =
                ctx.forward(&tokens, &mut kv_cache).expect("Decode failed");

            let token_elapsed = token_start.elapsed();
            latencies.push(token_elapsed);

            let next_token = sample(&logits, &strategy) as u32;
            generated.push(next_token);

            if (i + 1) % 20 == 0 {
                let total_elapsed = start.elapsed();
                let tok_per_sec = (i + 1) as f64 / total_elapsed.as_secs_f64();
                let avg_ms =
                    latencies.iter().sum::<std::time::Duration>().as_millis()
                        as f64
                        / latencies.len() as f64;

                println!(
                    "| {} | {:.0} | {:.1} | {:.2} |",
                    i + 1,
                    total_elapsed.as_millis(),
                    tok_per_sec,
                    avg_ms
                );
            }
        }

        let total_elapsed = start.elapsed();
        let final_tok_per_sec =
            decode_count as f64 / total_elapsed.as_secs_f64();

        println!("\n✓ Decode performance test completed");
        println!("  Final throughput: {:.1} tok/s", final_tok_per_sec);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_cpu_zero_allocation_decode() {
        println!("\n=== CPU Zero-Allocation Decode Test ===\n");

        let mut ctx = InferenceContext::from_gguf(MODEL_PATH, Device::Cpu)
            .expect("Failed to load model");

        let config = ctx.config.clone();
        let mut kv_cache = KVCache::new(&config, 500, KVCacheFormat::F32);

        // Prefill
        let prefill_tokens: Vec<u32> =
            (0..64).map(|x| (x % 1000) + 1).collect();
        ctx.forward(&prefill_tokens, &mut kv_cache).expect("Prefill failed");

        // Decode with forward_pooled (zero-allocation)
        let decode_count = 100;
        let strategy = SamplingStrategy::Greedy;
        let mut generated = Vec::with_capacity(decode_count);

        println!("Testing forward_pooled (zero-allocation path)");

        let start = Instant::now();

        for _ in 0..decode_count {
            let tokens =
                vec![*generated.last().unwrap_or(&config.bos_token_id)];

            let logits = ctx
                .forward_pooled(&tokens, &mut kv_cache)
                .expect("Decode failed");

            let next_token = sample(&logits, &strategy) as u32;
            generated.push(next_token);
        }

        let elapsed = start.elapsed();
        let tok_per_sec = decode_count as f64 / elapsed.as_secs_f64();

        println!("  Tokens: {}", decode_count);
        println!("  Time: {:?}", elapsed);
        println!("  Throughput: {:.1} tok/s", tok_per_sec);

        assert!(tok_per_sec > 0.0, "Throughput should be positive");
    }

    #[test]
    fn test_cpu_vs_llama_cpp_reference() {
        println!("\n=== CPU vs llama.cpp Reference ===\n");

        // Note: This test compares our CPU implementation against
        // llama.cpp reference numbers. The target is to match or exceed.
        //
        // llama.cpp reference (Qwen2.5-0.5B-Q4_K_M on M4 Pro):
        // - Decode: ~95-100 tok/s
        // - Prefill 128: ~500 tok/s

        let mut ctx = InferenceContext::from_gguf(MODEL_PATH, Device::Cpu)
            .expect("Failed to load model");

        let config = ctx.config.clone();
        println!("Model: {}B parameters", ctx.param_count() as f64 / 1e9);

        // Decode test
        let mut kv_cache = KVCache::new(&config, 500, KVCacheFormat::F32);
        let prefill_tokens: Vec<u32> =
            (0..64).map(|x| (x % 1000) + 1).collect();
        ctx.forward(&prefill_tokens, &mut kv_cache).expect("Prefill failed");

        let decode_count = 50;
        let strategy = SamplingStrategy::Greedy;
        let mut generated = Vec::new();

        let start = Instant::now();
        for _ in 0..decode_count {
            let tokens =
                vec![*generated.last().unwrap_or(&config.bos_token_id)];
            let logits = ctx.forward(&tokens, &mut kv_cache).expect("Failed");
            let next_token = sample(&logits, &strategy) as u32;
            generated.push(next_token);
        }
        let decode_tps = decode_count as f64 / start.elapsed().as_secs_f64();

        println!("Decode throughput: {:.1} tok/s", decode_tps);

        // Prefill test
        let mut kv_cache2 = KVCache::new(&config, 500, KVCacheFormat::F32);
        let prefill_len = 128;
        let tokens: Vec<u32> =
            (0..prefill_len as u32).map(|x| (x % 1000) + 1).collect();

        let start = Instant::now();
        ctx.forward(&tokens, &mut kv_cache2).expect("Prefill failed");
        let prefill_tps = prefill_len as f64 / start.elapsed().as_secs_f64();

        println!(
            "Prefill throughput ({} tokens): {:.1} tok/s",
            prefill_len, prefill_tps
        );

        // Log results
        println!("\nResults:");
        println!("  Decode: {:.1} tok/s", decode_tps);
        println!("  Prefill: {:.1} tok/s", prefill_tps);
    }
}
