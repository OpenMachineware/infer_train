// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Check model tensor dtypes

fn main() {
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string()
    });

    println!("=== Model Tensor Dtypes ===\n");

    use infer_train::model::weights::WeightStore;
    use infer_train::model::config::ModelConfig;

    let (store, config) = WeightStore::from_gguf(&model_path).unwrap();

    println!("Model: {:?}", config.name);
    println!("Hidden size: {}", config.hidden_size);
    println!("Layers: {}", config.n_layers);
    println!();

    // Check output tensor
    if let Some(output) = store.get("output.weight") {
        println!("Output tensor:");
        println!("  Shape: {:?}", output.shape);
        println!("  Dtype: {:?}", output.dtype);
        println!("  Nelements: {}", output.nelements);
    }

    // Check a few layer tensors
    println!("\nSample layer tensors:");
    for name in &["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.ffn_gate.weight"] {
        if let Some(tensor) = store.get(name) {
            println!("  {}: {:?} shape={:?}", name, tensor.dtype, tensor.shape);
        }
    }

    // Count dtypes
    println!("\nDtype distribution:");
    let mut dtype_counts = std::collections::HashMap::new();
    for (name, tensor) in store.iter() {
        if !name.contains("blk.") { continue; }
        *dtype_counts.entry(tensor.dtype).or_insert(0) += 1;
    }
    for (dtype, count) in dtype_counts.iter() {
        println!("  {:?}: {}", dtype, count);
    }
}
