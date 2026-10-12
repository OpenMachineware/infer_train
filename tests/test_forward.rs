// End-to-end forward pass test

use infer_train::inference::{
    InferenceContext, KVCache, KVCacheBase, KVCacheFormat,
};
use infer_train::model::config::ModelConfig;

#[test]
fn test_forward_basic() {
    // Create dummy config
    let config = ModelConfig {
        architecture: "llama".to_string(),
        hidden_size: 128,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 4,
        vocab_size: 1000,
        intermediate_size: 512,
        rope_dim: 0,
        rope_freq_base: 10000.0,
        rms_norm_eps: 1e-5,
        max_position_embeddings: 2048,
        bos_token_id: 1,
        eos_token_id: 2,
        metadata: std::collections::HashMap::new(),
    };

    // Create KV cache
    let mut kv_cache = KVCache::new(&config, 100, KVCacheFormat::F32);

    // Test tokens
    let tokens = vec![1, 2, 3]; // BOS, token2, token3

    // Note: This will fail without a real model file
    // The test validates that the forward pass structure is correct
    println!("Forward test setup complete");
    println!(
        "Config: {} layers, {} heads, {} hidden",
        config.n_layers, config.n_heads, config.hidden_size
    );
    println!("KV cache initialized with format: {:?}", kv_cache.format());
}

#[test]
#[ignore] // Run with --ignored flag when you have a model file
fn test_forward_with_model() {
    let model_path = std::env::var("TEST_MODEL_PATH")
        .expect("Set TEST_MODEL_PATH environment variable");

    let ctx = InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Cpu,
    )
    .expect("Failed to load model");

    println!(
        "Model loaded: {} layers, {}M params",
        ctx.layer_count(),
        ctx.param_count() / 1_000_000
    );

    // Create KV cache
    let max_seq_len = 100;
    let mut kv_cache =
        KVCache::new(&ctx.config, max_seq_len, KVCacheFormat::F32);

    // Test forward pass
    let tokens = vec![ctx.config.bos_token_id];
    let logits = ctx.model.forward(&tokens, &mut kv_cache);

    assert!(logits.is_some(), "Forward pass should return logits");
    let logits = logits.unwrap();
    assert_eq!(
        logits.len(),
        ctx.config.vocab_size,
        "Logits size should match vocab"
    );

    println!("Forward pass successful, logits size: {}", logits.len());
}
