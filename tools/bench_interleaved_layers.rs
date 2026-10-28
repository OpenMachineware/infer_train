// SPDX-License-Identifier: AGPL-3.0-or-later
// Benchmark interleaved GEMV across all layers to check cache thrashing

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
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model from {}...", model_path);

    // Load model
    let store =
        WeightStore::from_gguf(model_path).expect("Failed to load weights");
    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let raw_config = ModelConfig::from_gguf(&gguf);
    let model = LlamaModel::from_weights(&store, raw_config);

    let hidden_size = model.config.hidden_size;
    println!("hidden_size: {}", hidden_size);
    println!("n_layers: {}", model.layers.len());

    // Find Q4_K tensor type to test
    let layer0 = &model.layers[0];
    let tensor_type = if layer0.wq.as_ref().map(|t| t.dtype)
        == Some(infer_train::model::tensor::DataType::Q4_K)
    {
        "wq"
    } else if layer0.wo.as_ref().map(|t| t.dtype)
        == Some(infer_train::model::tensor::DataType::Q4_K)
    {
        "wo"
    } else if layer0.w1.as_ref().map(|t| t.dtype)
        == Some(infer_train::model::tensor::DataType::Q4_K)
    {
        "w1"
    } else {
        eprintln!("No Q4_K tensor found in layer 0");
        std::process::exit(1);
    };

    println!("Testing tensor: {}", tensor_type);

    // Create random input (same for all layers)
    let k = hidden_size;
    let nb = k / QK_K;
    let mut rng = rand::thread_rng();
    let x_f32: Vec<f32> =
        (0..k).map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0)).collect();
    let mut x_q8k = vec![unsafe { std::mem::zeroed() }; nb];
    quantize_row_q8_k_neon(&x_f32, &mut x_q8k);

    let n_iterations = 10;

    println!(
        "\nLayer | Standard (ms) | Interleaved (ms) | Speedup | Std BW | Interleaved BW"
    );
    println!(
        "------+---------------+-------------------+---------+--------+---------------"
    );

    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let test_tensor = match tensor_type {
            "wq" => layer.wq.as_ref(),
            "wo" => layer.wo.as_ref(),
            "w1" => layer.w1.as_ref(),
            _ => unreachable!(),
        };

        let test_tensor = match test_tensor {
            Some(t) => t,
            None => continue,
        };

        // Skip if not Q4_K
        if test_tensor.dtype != infer_train::model::tensor::DataType::Q4_K {
            continue;
        }

        let out_size = test_tensor.shape[1];
        let k = test_tensor.shape[0];
        let nb = k / QK_K;

        let weights: &[BlockQ4K] = unsafe {
            std::slice::from_raw_parts(
                test_tensor.bytes().as_ptr() as *const BlockQ4K,
                out_size * nb,
            )
        };

        // Standard GEMV
        let mut total_std = 0u128;
        for _ in 0..n_iterations {
            let start = Instant::now();
            let mut dst = vec![0.0f32; out_size];
            for row in 0..out_size {
                let row_weights: &[BlockQ4K] =
                    &weights[row * nb..(row + 1) * nb];
                unsafe {
                    dst[row] = vec_dot_q4_k_q8_k_neon(k, row_weights, &x_q8k);
                }
            }
            total_std += start.elapsed().as_nanos();
        }
        let std_avg = total_std as f64 / n_iterations as f64 / 1e6;

        // Interleaved GEMV
        let weights_interleaved = repack_q4_k_batch(weights, out_size, nb);
        let mut total_inter = 0u128;
        for _ in 0..n_iterations {
            let start = Instant::now();
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
            total_inter += start.elapsed().as_nanos();
        }
        let inter_avg = total_inter as f64 / n_iterations as f64 / 1e6;

        // Calculate bandwidth
        let weights_bytes = (out_size * k / 2) as f64;
        let input_bytes = (k * 4) as f64;
        let output_bytes = (out_size * 4) as f64;
        let total_bytes = weights_bytes + input_bytes + output_bytes;
        let std_bw = total_bytes / (std_avg / 1000.0) / 1e9;
        let inter_bw = total_bytes / (inter_avg / 1000.0) / 1e9;

        let speedup = std_avg / inter_avg;

        println!(
            "  {:2}  |    {:6.2}      |      {:6.2}        |  {:4.2}x   | {:5.1}  | {:5.1}",
            layer_idx, std_avg, inter_avg, speedup, std_bw, inter_bw
        );
    }
}
