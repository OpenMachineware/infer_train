// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug embedding shape and access

use infer_train::model::gguf::GgufFile;
use infer_train::model::weights::WeightStore;

const MODEL_PATH: &str =
    "/Users/jia/.llm_models/Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf";

fn main() {
    let _gguf = GgufFile::open(MODEL_PATH).expect("Failed to open GGUF");
    let store =
        WeightStore::from_gguf(MODEL_PATH).expect("Failed to load weights");

    // Find embedding tensor
    let embed = store
        .get("tok_embeddings.weight")
        .or_else(|| store.get("token_embd.weight"))
        .expect("No embedding tensor found");

    println!("Embedding tensor:");
    println!("  dtype: {:?}", embed.dtype);
    println!("  shape: {:?}", embed.shape);
    println!("  bytes len: {}", embed.bytes().len());

    // Calculate expected sizes
    let hidden_size = embed.shape.get(0).copied().unwrap_or(0);
    let vocab_size = embed.shape.get(1).copied().unwrap_or(0);
    println!("  hidden_size: {}", hidden_size);
    println!("  vocab_size: {}", vocab_size);

    // Check if shape is [vocab_size, hidden_size] or [hidden_size, vocab_size]
    // Try token 0 and token 22177
    let bytes = embed.bytes();
    let dtype = embed.dtype;

    // Calculate bytes per token based on different assumptions
    let bytes_per_elem_hidden = dtype.bytes_for_elems(hidden_size);
    let bytes_per_elem_vocab = dtype.bytes_for_elems(vocab_size);

    println!("\nBytes calculations:");
    println!("  bytes_for_elems(hidden_size): {}", bytes_per_elem_hidden);
    println!("  bytes_for_elems(vocab_size): {}", bytes_per_elem_vocab);

    // Check if offset calculation is correct
    // Token 22177 should be at offset 22177 * bytes_per_elem_hidden
    let offset = 22177 * bytes_per_elem_hidden;
    println!("\nToken 22177 offset: {}", offset);
    println!("  Total bytes: {}", bytes.len());
    println!(
        "  Would overflow: {}",
        offset + bytes_per_elem_hidden > bytes.len()
    );

    // Let's check the actual data at token 0 and token 1
    println!("\nToken 0 embedding (first 10 bytes):");
    let data = &bytes[0..10.min(bytes.len())];
    println!("  {:?}", data);

    // Check token 1
    if bytes_per_elem_hidden < bytes.len() {
        println!("\nToken 1 embedding (first 10 bytes):");
        let data = &bytes[bytes_per_elem_hidden
            ..bytes_per_elem_hidden
                + 10.min(bytes.len() - bytes_per_elem_hidden)];
        println!("  {:?}", data);
    }

    // Actually dequantize and check tokens
    println!("\n=== Dequantized embeddings ===");
    let dequant = infer_train::inference::dequant_to_f32;

    // Token 1
    let offset1 = 1 * bytes_per_elem_hidden;
    if offset1 + bytes_per_elem_hidden <= bytes.len() {
        let emb1 = dequant(
            &bytes[offset1..offset1 + bytes_per_elem_hidden],
            dtype,
            hidden_size,
        );
        let num_nan = emb1.iter().filter(|&&v| v.is_nan()).count();
        let max = emb1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = emb1.iter().cloned().fold(f32::INFINITY, f32::min);
        println!("Token 1: nan={}, range=[{:.4}, {:.4}]", num_nan, min, max);
    }

    // Token 22177
    let offset22177 = 22177 * bytes_per_elem_hidden;
    if offset22177 + bytes_per_elem_hidden <= bytes.len() {
        let emb22177 = dequant(
            &bytes[offset22177..offset22177 + bytes_per_elem_hidden],
            dtype,
            hidden_size,
        );
        let num_nan = emb22177.iter().filter(|&&v| v.is_nan()).count();
        let max = emb22177.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = emb22177.iter().cloned().fold(f32::INFINITY, f32::min);
        println!(
            "Token 22177: nan={}, range=[{:.4}, {:.4}]",
            num_nan, min, max
        );
        println!("  First 5 values: {:?}", &emb22177[0..5.min(emb22177.len())]);
    }

    // Token 4304
    let offset4304 = 4304 * bytes_per_elem_hidden;
    if offset4304 + bytes_per_elem_hidden <= bytes.len() {
        let emb4304 = dequant(
            &bytes[offset4304..offset4304 + bytes_per_elem_hidden],
            dtype,
            hidden_size,
        );
        let num_nan = emb4304.iter().filter(|&&v| v.is_nan()).count();
        let max = emb4304.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = emb4304.iter().cloned().fold(f32::INFINITY, f32::min);
        println!("Token 4304: nan={}, range=[{:.4}, {:.4}]", num_nan, min, max);
    }
}
