// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test CPU inference

use infer_train::model::gguf::GgufFile;
use infer_train::model::llama::LlamaModel;
use infer_train::model::weights::WeightStore;

const MODEL_PATH: &str =
    "/Users/jia/.llm_models/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf";

#[test]
fn test_cpu_forward() {
    // Load GGUF
    let gguf = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF");
    let config = infer_train::model::config::ModelConfig::from_gguf(&gguf);
    let weights =
        WeightStore::from_gguf(MODEL_PATH).expect("Failed to load weights");
    let model = LlamaModel::from_weights(&weights, config.clone());

    println!(
        "Config: hidden={}, heads={}, kv_heads={}, head_dim={}",
        config.hidden_size,
        config.n_heads,
        config.n_kv_heads,
        model.config.head_dim
    );
    println!(
        "Model config: vocab_size={}, n_layers={}",
        model.config.vocab_size, model.config.n_layers
    );

    // Create CPU KV cache
    let mut kv_cache = infer_train::inference::KVCache::new(
        &model.config,
        100,
        infer_train::inference::KVCacheFormat::F32,
    );

    // Run forward pass with simple tokens
    let tokens = vec![1u32, 2, 3];
    println!("Tokens: {:?}", tokens);

    let start = std::time::Instant::now();
    let logits = model.forward(&tokens, &mut kv_cache).expect("Forward failed");
    let elapsed = start.elapsed();

    println!("Forward time: {:?}", elapsed);
    println!(
        "Logits: {} values (vocab_size={})",
        logits.len(),
        model.config.vocab_size
    );

    // Check for NaN
    let num_nan = logits.iter().filter(|&&v| v.is_nan()).count();
    let num_inf = logits.iter().filter(|&&v| v.is_infinite()).count();
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = logits.iter().cloned().fold(f32::INFINITY, f32::min);

    println!(
        "Stats: nan={}, inf={}, min={:.2}, max={:.2}",
        num_nan, num_inf, min_val, max_val
    );

    // Sample top token
    let max_idx = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i);
    println!("Top token: {:?}", max_idx);

    assert!(num_nan == 0, "Logits should not contain NaN");
    assert!(logits.len() == model.config.vocab_size);
}
