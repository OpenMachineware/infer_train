// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug NaN in GPU batch prefill

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};
use infer_train::model::gguf::GgufFile;
use infer_train::model::llama::LlamaModel;
use infer_train::model::weights::WeightStore;

const MODEL_PATH: &str =
    "/Users/jia/.llm_models/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf";

fn main() {
    // Load model
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

    // Create CPU KV cache
    let mut kv_cache_cpu = KVCache::new(&model.config, 100, KVCacheFormat::F32);

    // Test tokens - use actual tokens from "Hello world"
    let tokens = vec![1u32, 22177, 4304]; // "Hello world" tokens
    println!("Tokens: {:?}", tokens);

    // Run CPU forward
    println!("\n=== CPU Forward ===");
    let logits_cpu =
        model.forward(&tokens, &mut kv_cache_cpu).expect("CPU forward failed");
    let num_nan_cpu = logits_cpu.iter().filter(|&&v| v.is_nan()).count();
    let max_cpu = logits_cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_cpu = logits_cpu.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "CPU: nan={}, range=[{:.4}, {:.4}]",
        num_nan_cpu, min_cpu, max_cpu
    );

    // Run GPU forward
    println!("\n=== GPU Forward ===");
    let mut ctx = InferenceContext::from_gguf(MODEL_PATH, Device::Gpu)
        .expect("Failed to load GPU context");

    let logits_gpu =
        ctx.forward_prefill_batch(&tokens).expect("GPU forward failed");
    let num_nan_gpu = logits_gpu.iter().filter(|&&v| v.is_nan()).count();
    let max_gpu = logits_gpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_gpu = logits_gpu.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "GPU: nan={}, range=[{:.4}, {:.4}]",
        num_nan_gpu, min_gpu, max_gpu
    );

    // Check if both produce similar results (when no NaN)
    if num_nan_gpu == 0 {
        let diff: f32 = logits_cpu
            .iter()
            .zip(logits_gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / logits_cpu.len() as f32;
        println!("Mean absolute difference: {:.6}", diff);
    }
}
