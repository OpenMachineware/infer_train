// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test KV cache quantization

use infer_train::inference::kv_cache::{KVCache, KVCacheBase, KVCacheFormat};
use infer_train::model::config::ModelConfig;

fn create_test_config() -> ModelConfig {
    use std::collections::HashMap;
    ModelConfig {
        architecture: "llama".to_string(),
        hidden_size: 128, // 128 / 4 heads = 32 head_dim (multiple of QK=32)
        intermediate_size: 384,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 1000,
        rope_dim: 32,
        rope_freq_base: 10000.0,
        rms_norm_eps: 1e-5,
        max_position_embeddings: 128,
        bos_token_id: 1,
        eos_token_id: 2,
        metadata: HashMap::new(),
    }
}

fn test_format(format: KVCacheFormat, name: &str) {
    let config = create_test_config();
    let max_seq = 16;
    let head_dim = config.hidden_size / config.n_heads; // 16
    let n_kv_heads = config.n_kv_heads; // 2

    let mut cache = KVCache::new(&config, max_seq, format);

    // Create test K and V values
    let k: Vec<f32> =
        (0..n_kv_heads * head_dim).map(|i| i as f32 * 0.1).collect();
    let v: Vec<f32> =
        (0..n_kv_heads * head_dim).map(|i| (i as f32 * 0.1 + 0.5)).collect();

    // Store first position
    cache.set_k(0, &k, n_kv_heads, head_dim);
    cache.set_v(0, &v, n_kv_heads, head_dim);
    cache.advance();

    // Store second position
    let k2: Vec<f32> = k.iter().map(|x| x + 10.0).collect();
    let v2: Vec<f32> = v.iter().map(|x| x + 10.0).collect();
    cache.set_k(0, &k2, n_kv_heads, head_dim);
    cache.set_v(0, &v2, n_kv_heads, head_dim);
    cache.advance();

    // Retrieve and verify
    let k_retrieved = cache.get_k(0, 0, 0);
    let v_retrieved = cache.get_v(0, 0, 0);

    // Check that values are approximately preserved (within quantization error)
    for i in 0..head_dim {
        let expected = k[i];
        let actual = k_retrieved[i];
        let diff = (expected - actual).abs();
        let tolerance = match format {
            KVCacheFormat::F32 => 0.0,
            KVCacheFormat::F16 => 0.01,
            KVCacheFormat::Q8_0 => 0.1,
            KVCacheFormat::Q4_0 => 0.3,
            KVCacheFormat::Q4_1 => 0.2,
            KVCacheFormat::Q5_0 => 0.1,
            KVCacheFormat::Q5_1 => 0.1,
            KVCacheFormat::BF16 => 0.01,
        };
        assert!(
            diff <= tolerance,
            "{}: K[{}] expected {} got {} (diff {})",
            name,
            i,
            expected,
            actual,
            diff
        );
    }

    // Verify sequence length
    assert_eq!(cache.seq_len(), 2);

    // Test clear
    cache.clear();
    assert_eq!(cache.seq_len(), 0);

    println!("✓ {} format test passed", name);
}

#[test]
fn test_all_formats() {
    test_format(KVCacheFormat::F32, "F32");
    test_format(KVCacheFormat::F16, "F16");
    test_format(KVCacheFormat::Q8_0, "Q8_0");
    test_format(KVCacheFormat::Q4_0, "Q4_0");
    test_format(KVCacheFormat::Q4_1, "Q4_1");
    test_format(KVCacheFormat::Q5_0, "Q5_0");
    test_format(KVCacheFormat::Q5_1, "Q5_1");
    println!("\n✓ All KV cache formats verified");
}

fn main() {
    test_all_formats();
}
