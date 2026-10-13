// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test single forward pass

#[cfg(test)]
mod tests {
    use infer_train::inference::{InferenceContext, KVCache, KVCacheFormat};

    #[test]
    fn test_forward_pass() {
        let model_path = std::env::var("TEST_MODEL")
            .unwrap_or_else(|_| {
                "/Users/jia/.llm_models/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf".to_string()
            });

        println!("Loading model: {}", model_path);

        let ctx = InferenceContext::from_gguf(
            &model_path,
            infer_train::inference::Device::Cpu,
        )
        .expect("Failed to load model");

        println!("✓ Model loaded");
        println!("  Layers: {}", ctx.model.layers.len());

        // Test single forward pass
        println!("\n=== Testing Forward Pass ===");

        let config = ctx.config.clone();
        let mut kv_cache = KVCache::new(&config, 10, KVCacheFormat::F32);

        // Use tokens 1, 2, 3 (non-zero embeddings)
        let tokens: Vec<u32> = vec![1, 2, 3];

        println!("Input tokens: {:?}", tokens);

        // Process token by token
        for (i, &token) in tokens.iter().enumerate() {
            println!("\nProcessing token {} (value {}):", i, token);

            // Get embedding
            if let Some(embed) = ctx.model.test_embedding_lookup(token) {
                let non_zero = embed.iter().filter(|&&v| v != 0.0).count();
                println!(
                    "  Embedding: {} non-zero / {}",
                    non_zero,
                    embed.len()
                );
            } else {
                println!("  ERROR: Failed to get embedding");
                continue;
            }

            // Run forward pass
            match ctx.model.forward(&[token], &mut kv_cache) {
                Some(logits) => {
                    let non_zero = logits.iter().filter(|&&v| v != 0.0).count();
                    let max_idx = logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i);
                    let max_val = logits
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);

                    println!(
                        "  Logits: {} values, {} non-zero",
                        logits.len(),
                        non_zero
                    );
                    println!("  Max logit: {} at index {:?}", max_val, max_idx);
                }
                None => {
                    println!("  ERROR: Forward pass returned None");
                }
            }

            kv_cache.advance();
        }

        println!("\n=== Done ===");
    }
}
