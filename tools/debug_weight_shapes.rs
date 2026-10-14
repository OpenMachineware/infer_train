// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug weight shapes

use infer_train::model::gguf::GgufFile;
use infer_train::model::weights::WeightStore;

const MODEL_PATH: &str =
    "/Users/jia/.llm_models/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf";

fn main() {
    let gguf = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF");
    let config = infer_train::model::config::ModelConfig::from_gguf(&gguf);

    println!("Config from metadata:");
    println!("  hidden_size: {}", config.hidden_size);
    println!("  n_heads: {}", config.n_heads);
    println!("  n_kv_heads: {}", config.n_kv_heads);
    println!("  head_dim: {}", config.head_dim);
    println!("  intermediate_size: {}", config.intermediate_size);
    println!("  n_layers: {}", config.n_layers);

    let store =
        WeightStore::from_gguf(MODEL_PATH).expect("Failed to load weights");

    // Print first 20 weight keys
    println!("\nFirst 20 weight keys:");
    for (i, key) in store.keys().take(20).enumerate() {
        if let Some(tensor) = store.get(key) {
            println!(
                "  {}: {} -> dtype={:?}, shape={:?}",
                i, key, tensor.dtype, tensor.shape
            );
        }
    }

    // Check if we can find layer 0 weights
    println!("\nSearching for layer 0 weights...");
    for key in store.keys() {
        if key.contains("layers.0") || key.contains("blk.0") {
            if let Some(tensor) = store.get(key) {
                println!(
                    "  Found: {} -> dtype={:?}, shape={:?}",
                    key, tensor.dtype, tensor.shape
                );
            }
        }
    }
}
