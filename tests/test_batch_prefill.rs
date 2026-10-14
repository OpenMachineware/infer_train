// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test batch prefill implementation

#![allow(clippy::approx_constant)]

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
mod tests {
    use infer_train::inference::KVCacheGPU;
    use infer_train::model::gguf::GgufFile;
    use infer_train::model::llama::LlamaModel;
    use infer_train::model::weights::WeightStore;
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;

    const MODEL_PATH: &str =
        "/Users/jia/.llm_models/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf";

    /// Test batch prefill with Devstral Q4_K_M model
    #[test]
    fn test_batch_prefill_devstral() {
        // Load GGUF file
        let gguf = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF");

        // Load model config and weights
        let config = infer_train::model::config::ModelConfig::from_gguf(&gguf);
        println!(
            "GGUF config: hidden_size={}, n_heads={}, n_kv_heads={}, head_dim={}, n_layers={}",
            config.hidden_size,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.n_layers
        );

        let weights =
            WeightStore::from_gguf(MODEL_PATH).expect("Failed to load model");
        let model = LlamaModel::from_weights(&weights, config.clone());

        // Print fixed config
        println!(
            "Model config (fixed): hidden_size={}, n_heads={}, n_kv_heads={}, head_dim={}",
            model.config.hidden_size,
            model.config.n_heads,
            model.config.n_kv_heads,
            model.config.head_dim
        );

        // Check actual weight shapes
        if let Some(layer) = model.layers.first() {
            if let (Some(wq), Some(wk), Some(wv)) =
                (&layer.wq, &layer.wk, &layer.wv)
            {
                println!(
                    "wq shape: {:?}, wk shape: {:?}, wv shape: {:?}",
                    wq.shape, wk.shape, wv.shape
                );
            }
        }

        // Create GPU context
        let gpu_ctx =
            MetalContext::new().expect("Failed to create MetalContext");

        // Test batch prefill with different batch sizes
        for batch_size in [1, 4, 8, 16].iter() {
            // Create a fresh KV cache for each test using model's fixed config
            let mut kv_cache =
                KVCacheGPU::new(gpu_ctx.device(), &model.config, 4096);
            let tokens: Vec<u32> = (0..*batch_size as u32).collect();

            let result =
                model.forward_prefill_batch(&tokens, &mut kv_cache, &gpu_ctx);

            if let Some(logits) = result {
                println!(
                    "Batch size {}: got {} logits",
                    batch_size,
                    logits.len()
                );
                assert!(logits.len() == model.config.vocab_size);
            } else {
                println!(
                    "Batch size {}: forward_prefill_batch returned None",
                    batch_size
                );
            }
        }

        println!("Batch prefill test completed");
    }
}
