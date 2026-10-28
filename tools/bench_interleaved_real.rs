// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Compare standard vs interleaved GEMV performance on real model weights

use infer_train::model::weights::WeightStore;
use infer_train::model::{GgufFile, LlamaModel, ModelConfig};
use infer_train::quant::kv_cache::quant_q8k::quantize_row_q8_k_neon;
use infer_train::quant::types::{BlockQ4K, BlockQ8K, QK_K};
use infer_train::quant::vec_dot::arm::{
    repack_q4_k_batch, vec_dot_q4_k_q8_k_interleaved, vec_dot_q4_k_q8_k_neon,
};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf> [tensor_type]", args[0]);
        eprintln!("  tensor_type: wq, wk, wo, w1, w2, w3 (default: w1)");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let tensor_type = args.get(2).map(|s| s.as_str()).unwrap_or("w1");

    println!("Loading model from {}...", model_path);

    // Load model
    let store =
        WeightStore::from_gguf(model_path).expect("Failed to load weights");
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let raw_config = ModelConfig::from_gguf(&gguf);
    let model = LlamaModel::from_weights(&store, raw_config);

    let hidden_size = model.config.hidden_size;
    println!("hidden_size: {}", hidden_size);

    // Find a Q4_K tensor to test
    let layer = &model.layers[0];
    let test_tensor = match tensor_type {
        "wq" => layer.wq.as_ref(),
        "wk" => layer.wk.as_ref(),
        "wo" => layer.wo.as_ref(),
        "w1" => layer.w1.as_ref(),
        "w2" => layer.w2.as_ref(),
        "w3" => layer.w3.as_ref(),
        _ => {
            eprintln!("Unknown tensor type: {}", tensor_type);
            std::process::exit(1);
        }
    }
    .expect("Tensor not found");

    println!("Using tensor: {} (dtype: {:?})", tensor_type, test_tensor.dtype);

    // Check if Q4_K
    if test_tensor.dtype != infer_train::model::tensor::DataType::Q4_K {
        eprintln!("Error: Tensor is not Q4_K format, skipping benchmark");
        std::process::exit(1);
    }

    let out_size = test_tensor.shape[1];
    let k = test_tensor.shape[0];
    let nb = k / QK_K;

    println!("\nTest tensor: {} x {} ({} blocks)", k, out_size, nb);
    println!("Tensor dtype: {:?}", test_tensor.dtype);

    // Get weight data
    let weights: &[BlockQ4K] = unsafe {
        std::slice::from_raw_parts(
            test_tensor.bytes().as_ptr() as *const BlockQ4K,
            out_size * nb,
        )
    };

    // Create random input
    let mut rng = rand::thread_rng();
    let x_f32: Vec<f32> =
        (0..k).map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0)).collect();

    // Quantize input to Q8_K
    let mut x_q8k = vec![unsafe { std::mem::zeroed() }; nb];
    quantize_row_q8_k_neon(&x_f32, &mut x_q8k);

    println!("\n=== Standard GEMV ===");

    // Standard approach: process each row independently
    let n_warmup = 3;
    let n_iterations = 20;

    // Warmup
    for _ in 0..n_warmup {
        let mut dst = vec![0.0f32; out_size];
        for row in 0..out_size {
            let row_weights: &[BlockQ4K] = &weights[row * nb..(row + 1) * nb];
            unsafe {
                dst[row] = vec_dot_q4_k_q8_k_neon(k, row_weights, &x_q8k);
            }
        }
    }

    // Benchmark
    let mut total_time = 0u128;
    let mut dst_standard = vec![0.0f32; out_size];

    for _ in 0..n_iterations {
        let start = Instant::now();
        for row in 0..out_size {
            let row_weights: &[BlockQ4K] = &weights[row * nb..(row + 1) * nb];
            unsafe {
                dst_standard[row] =
                    vec_dot_q4_k_q8_k_neon(k, row_weights, &x_q8k);
            }
        }
        total_time += start.elapsed().as_nanos();
    }

    let standard_avg = total_time as f64 / n_iterations as f64 / 1e6;
    let standard_throughput =
        (out_size as f64 * k as f64) / (standard_avg / 1000.0) / 1e6;

    println!("Avg time: {:.2} ms", standard_avg);
    println!("Throughput: {:.2} M elements/s", standard_throughput);

    // Memory bandwidth
    let weights_bytes = (out_size * k / 2) as f64; // Q4_K = 4 bits
    let input_bytes = (k * 4) as f64;
    let output_bytes = (out_size * 4) as f64;
    let total_bytes = weights_bytes + input_bytes + output_bytes;
    let standard_bw = total_bytes / (standard_avg / 1000.0) / 1e9;
    println!("Bandwidth: {:.2} GB/s", standard_bw);

    println!("\n=== Interleaved GEMV ===");

    // Repack weights to interleaved format
    println!("Repacking weights...");
    let start = Instant::now();
    let weights_interleaved = repack_q4_k_batch(weights, out_size, nb);
    let repack_time = start.elapsed();
    println!("Repack time: {:.2} ms", repack_time.as_secs_f64() * 1000.0);

    // Warmup
    for _ in 0..n_warmup {
        let mut dst = vec![0.0f32; out_size];
        unsafe {
            vec_dot_q4_k_q8_k_interleaved(
                &weights_interleaved,
                &x_q8k,
                &mut dst,
                k,
                out_size,
            );
        }
    }

    // Benchmark
    let mut total_time = 0u128;
    let mut dst_interleaved = vec![0.0f32; out_size];

    for _ in 0..n_iterations {
        let start = Instant::now();
        unsafe {
            vec_dot_q4_k_q8_k_interleaved(
                &weights_interleaved,
                &x_q8k,
                &mut dst_interleaved,
                k,
                out_size,
            );
        }
        total_time += start.elapsed().as_nanos();
    }

    let interleaved_avg = total_time as f64 / n_iterations as f64 / 1e6;
    let interleaved_throughput =
        (out_size as f64 * k as f64) / (interleaved_avg / 1000.0) / 1e6;

    println!("Avg time: {:.2} ms", interleaved_avg);
    println!("Throughput: {:.2} M elements/s", interleaved_throughput);

    let interleaved_bw = total_bytes / (interleaved_avg / 1000.0) / 1e9;
    println!("Bandwidth: {:.2} GB/s", interleaved_bw);

    // Correctness check
    println!("\n=== Correctness ===");
    let mut max_rel_err = 0.0f32;
    let mut max_abs_err = 0.0f32;
    for i in 0..out_size.min(1000) {
        let abs_err = (dst_standard[i] - dst_interleaved[i]).abs();
        let rel_err = if dst_standard[i].abs() > 1e-6 {
            abs_err / dst_standard[i].abs()
        } else {
            abs_err
        };
        max_abs_err = max_abs_err.max(abs_err);
        max_rel_err = max_rel_err.max(rel_err);
    }
    println!("Max abs error: {:.6e}", max_abs_err);
    println!(
        "Max rel error: {:.6e} ({:.4}%)",
        max_rel_err,
        max_rel_err * 100.0
    );

    // Comparison
    println!("\n=== Summary ===");
    let speedup = standard_avg / interleaved_avg;
    println!("Speedup: {:.2}x", speedup);

    if speedup > 1.0 {
        println!("✓ Interleaved is {:.1}% faster", (speedup - 1.0) * 100.0);
    } else {
        println!("✗ Interleaved is {:.1}% slower", (1.0 - speedup) * 100.0);
    }
}
