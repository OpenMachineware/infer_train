// Test sequential vs parallel GEMV to understand cache effects

use infer_train::model::gguf::GgufWeightStore;
use infer_train::model::llama::LlamaModel;
use std::fs::File;
use std::io::BufReader;

fn main() {
    let model_path = std::env::args().nth(1).expect("Model path required");
    println!("Loading model from {}...", model_path);

    // Load weights
    let file = File::open(&model_path).expect("Failed to open file");
    let reader = BufReader::new(file);
    let weight_store =
        GgufWeightStore::from_file(reader).expect("Failed to parse GGUF");
    let model =
        LlamaModel::from_weights(&weight_store, weight_store.header().into());

    println!("\n=== Testing Sequential vs Parallel GEMV ===\n");

    // Get Q projection weight for each layer
    let n_layers = 28;

    // Test sequential execution (one layer at a time)
    println!("Sequential execution (layer by layer):");
    let mut total_time_ms = 0.0;

    for layer in 0..n_layers {
        let w = model
            .get_weight(&format!("blk.{}.attn_q.weight", layer))
            .expect("Weight not found");
        let in_size = w.shape()[1];
        let out_size = w.shape()[0];

        // Create test input
        let x: Vec<f32> =
            (0..in_size).map(|i| (i as f32 * 0.01).sin()).collect();

        // Time the GEMV
        let start = std::time::Instant::now();
        let _ = model.gemv_impl(&w, &x, in_size, out_size);
        let time_ms = start.elapsed().as_micros() as f64 / 1000.0;

        println!("  Layer {}: {:.2} ms", layer, time_ms);
        total_time_ms += time_ms;
    }

    println!("  Total: {:.2} ms\n", total_time_ms);

    // Test parallel execution (all layers at once using Rayon)
    println!("Parallel execution (all layers simultaneously using Rayon):");
    use rayon::prelude::*;

    let layer_weights: Vec<_> = (0..n_layers)
        .map(|layer| {
            let w = model
                .get_weight(&format!("blk.{}.attn_q.weight", layer))
                .unwrap();
            let in_size = w.shape()[1];
            let out_size = w.shape()[0];
            (w, in_size, out_size)
        })
        .collect();

    let x: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();

    let start = std::time::Instant::now();
    let results: Vec<_> = layer_weights
        .par_iter()
        .map(|(w, in_size, out_size)| {
            let x_layer: Vec<f32> =
                (0..*in_size).map(|i| (i as f32 * 0.01).sin()).collect();
            model.gemv_impl(w, &x_layer, *in_size, *out_size)
        })
        .collect();
    let time_ms = start.elapsed().as_micros() as f64 / 1000.0;

    println!("  Total: {:.2} ms", time_ms);
    println!("  Results: {} layers processed", results.len());

    // Test with thread-local buffers (avoid allocation overhead)
    println!("\nParallel with pre-allocated buffers:");
    let start = std::time::Instant::now();

    // Pre-allocate all inputs
    let inputs: Vec<Vec<f32>> = (0..n_layers)
        .map(|layer| {
            let in_size = layer_weights[layer].1;
            (0..in_size).map(|i| (i as f32 * 0.01).sin()).collect()
        })
        .collect();

    let alloc_time_ms = start.elapsed().as_micros() as f64 / 1000.0;
    println!("  Allocation time: {:.2} ms", alloc_time_ms);

    let start = std::time::Instant::now();
    let results: Vec<_> = layer_weights
        .par_iter()
        .zip(inputs.par_iter())
        .map(|((w, in_size, out_size), x)| {
            model.gemv_impl(w, x, *in_size, *out_size)
        })
        .collect();
    let compute_time_ms = start.elapsed().as_micros() as f64 / 1000.0;

    println!("  Compute time: {:.2} ms", compute_time_ms);
}
