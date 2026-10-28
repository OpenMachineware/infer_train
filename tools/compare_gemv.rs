// SPDX-License-Identifier: AGPL-3.0-or-later
// Compare GEMV implementations

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model from {}...", model_path);

    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    println!("Model config:");
    println!("  Hidden size: {}", ctx.config.hidden_size);
    println!("  Layers: {}", ctx.config.n_layers);

    // Get first layer's gate weight (w1) - should have shape [hidden_size, intermediate_size]
    let layer = &ctx.model.layers[0];
    let gate_weight = layer.w1.as_ref().expect("No gate weight");

    println!("\nGate weight:");
    println!("  Shape: {:?}", gate_weight.shape);
    println!("  Dtype: {:?}", gate_weight.dtype);

    // Create test input
    let hidden_size = ctx.config.hidden_size;
    let x: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.001).collect();

    // Test 1: Standard GEMV (gemv_impl -> matmul -> vec_dot)
    println!("\n=== Standard GEMV (gemv_impl) ===");
    let intermediate_size = gate_weight.shape[1];
    let result1 = ctx.model.gemv(&x, gate_weight, intermediate_size).unwrap();
    println!("  Output size: {}", result1.len());
    println!("  First 5: {:?}", &result1[..5]);

    // Test 2: Parallel row GEMV (gemv_parallel_row)
    println!("\n=== Parallel row GEMV (gemv_parallel_row) ===");
    let mut result2 = vec![0.0f32; intermediate_size];

    // Simulate thread 0 of 1 (single-threaded parallel)
    ctx.model.gemv_parallel_row(
        0,
        1,
        &x,
        gate_weight,
        &mut result2,
        hidden_size,
        intermediate_size,
    );
    println!("  Output size: {}", result2.len());
    println!("  First 5: {:?}", &result2[..5]);

    // Compare
    println!("\n=== Comparison ===");
    let mut max_diff = 0.0f32;
    let mut total_diff = 0.0f32;
    for i in 0..result1.len().min(result2.len()) {
        let diff = (result1[i] - result2[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        total_diff += diff;
    }
    let avg_diff = total_diff / result1.len() as f32;
    println!("  Max difference: {:.10}", max_diff);
    println!("  Avg difference: {:.10}", avg_diff);
}
