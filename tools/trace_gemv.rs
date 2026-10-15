// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Trace GEMV computation to debug matrix layout issues

use infer_train::inference::dequant_to_f32;
use infer_train::model::{
    GgufFile, ModelConfig, llama::LlamaModel, weights::WeightStore,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return;
    }

    let model_path = &args[1];
    let store = WeightStore::from_gguf(model_path).unwrap();
    let gguf = GgufFile::open(model_path).unwrap();
    let raw_config = ModelConfig::from_gguf(&gguf);
    let model = LlamaModel::from_weights(&store, raw_config);

    // Get layer 0 w1 weight
    let layer = &model.layers[0];
    let w1 = layer.w1.as_ref().unwrap();

    println!("=== Layer 0 FFN Gate Weight ===");
    println!("Shape: {:?}", w1.shape);
    println!("Dtype: {:?}", w1.dtype);
    println!("Bytes: {}", w1.bytes().len());

    // Dequantize the weight
    let hidden_size = w1.shape[0];
    let intermediate_size = w1.shape[1];
    let total_elems = hidden_size * intermediate_size;

    let w_data = dequant_to_f32(w1.bytes(), w1.dtype, total_elems);

    // Check values
    println!("\nFirst 10 values:");
    for i in 0..10 {
        println!("  w_data[{}] = {}", i, w_data[i]);
    }

    // Check if values are reasonable
    let max_val = w_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = w_data.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("\nOverall range: [{}, {}]", min_val, max_val);

    // Test GEMV with a simple input
    let x: Vec<f32> = vec![0.1; hidden_size];
    let result = model.gemv(&x, w1, intermediate_size).unwrap();

    println!("\n=== GEMV Test ===");
    println!("Input: {} values of 0.1", hidden_size);
    println!("Output: {} values", intermediate_size);
    println!("First 10 output values:");
    for i in 0..10 {
        println!("  out[{}] = {}", i, result[i]);
    }

    // Check output range
    let out_max = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let out_min = result.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("Output range: [{}, {}]", out_min, out_max);

    // Manually compute first output element to verify
    // In column-major with shape [K, M] = [in_size, out_size], element (row, col) = (i, j) is at i + j * K
    // For GEMV y = W @ x where W is shape [in_size, out_size]:
    // y[j] = sum_i W[i, j] * x[i]
    // W[i, j] is at index i + j * in_size

    // Compute y[0]: sum over i of W[i, 0] * x[i]
    let mut correct_sum = 0.0f32;
    for i in 0..hidden_size {
        correct_sum += w_data[i + 0 * hidden_size] * x[i];
    }
    println!("\nCorrect computation of out[0]: {}", correct_sum);
    println!("Actual out[0]: {}", result[0]);

    // Now compute what the GEMV is actually computing:
    // The code does: out[i] = sum_j w_data[j + i * in_size] * x[j]
    // For i=0: out[0] = sum_j w_data[j + 0] * x[j]
    let mut gemv_computed = 0.0f32;
    for j in 0..hidden_size {
        gemv_computed += w_data[j + 0 * hidden_size] * x[j];
    }
    println!("What GEMV computes: {}", gemv_computed);

    // Both should be the same if the index is j + i * in_size
    // Let me try the TRANSPOSED version:
    // y[i] = sum_j W[j, i] * x[j]
    // W[j, i] is at index j + i * in_size
    let mut transposed_sum = 0.0f32;
    for j in 0..hidden_size {
        transposed_sum += w_data[j + 0 * hidden_size] * x[j];
    }
    println!("Transposed computation: {}", transposed_sum);

    // Check if they match
    if (correct_sum - result[0]).abs() > 0.001 {
        println!("\nMISMATCH!");
        println!("Expected (correct): {}", correct_sum);
        println!("Actual (GEMV): {}", result[0]);

        // The GEMV code computes out[i] = sum_j w_data[j + i * in_size] * x[j]
        // This is sum_j W[j, i] * x[j]
        // Which is: y = W^T @ x
        // But we want: y = W @ x, which is y[i] = sum_j W[i, j] * x[j]
        // W[i, j] is at j + i * in_size in column-major
        println!("\nThe GEMV is computing W^T @ x, but we need W @ x!");
        println!(
            "Need to swap the index to w_data[i + j * in_size] or j + i * K"
        );
    }
}
