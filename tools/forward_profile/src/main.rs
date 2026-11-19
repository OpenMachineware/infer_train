// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Detailed forward pass profiling tool
//!
//! Measures absolute time for each operation by instrumenting forward.rs

use clap::Parser;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "forward_profile")]
#[command(about = "Detailed forward pass profiling")]
struct Cli {
    /// Model path
    #[arg(short, long)]
    model: String,

    /// Prompt for testing
    #[arg(short, long, default_value = "Hello")]
    prompt: String,

    /// Number of decode tokens
    #[arg(long, default_value = "10")]
    decode_tokens: usize,

    /// Number of warmup iterations
    #[arg(long, default_value = "2")]
    warmup: usize,

    /// Number of measurement iterations
    #[arg(long, default_value = "5")]
    iterations: usize,

    /// Thread count for decode (0 = auto, llama.cpp style adaptive)
    #[arg(long, default_value = "0")]
    threads: usize,

    /// Thread count for prefill (0 = auto, llama.cpp style adaptive)
    #[arg(long, default_value = "0")]
    threads_batch: usize,
}

fn main() {
    let cli = Cli::parse();

    println!("=== Forward Profile ===");
    println!("Model: {}", cli.model);
    println!("Prompt: {}", cli.prompt);
    println!("Decode tokens: {}", cli.decode_tokens);
    println!("Iterations: {}", cli.iterations);
    println!();

    // Load model
    use infer_train::inference::ForwardExecutor;
    use infer_train::inference::SimpleModel;
    use infer_train::model::GgufFile;
    use infer_train::tokenizer::create_tokenizer;

    let gguf = GgufFile::load(&cli.model).expect("Failed to load model");
    println!("Quantization types in model:");

    // Count quant types
    let mut type_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for t in &gguf.tensors {
        let type_name = format!("{:?}", t.quant_type);
        *type_counts.entry(type_name).or_insert(0) += 1;
    }
    for (t, count) in &type_counts {
        println!("  {}: {} tensors", t, count);
    }

    let tokenizer = create_tokenizer(
        gguf.tokenizer_metadata().expect("Failed to get metadata"),
    )
    .expect("Failed to create tokenizer");
    let model =
        SimpleModel::load(&cli.model, tokenizer).expect("Failed to load model");

    let config = model.config().clone();
    let n_layers = config.n_layers;
    println!("\nModel config:");
    println!("  Layers: {}", n_layers);
    println!("  Hidden size: {}", config.hidden_size);
    println!("  Intermediate size: {:?}", config.intermediate_size);
    println!("  Heads: {}", config.n_heads);
    println!("  KV heads: {}", config.n_kv_heads);

    // Get tensor info for multiple linear layers to understand layout
    println!("\n=== Weight Layout Analysis ===");
    for t in &gguf.tensors {
        if t.name.contains("attn_q.weight")
            || t.name.contains("ffn_gate.weight")
        {
            println!("\nTensor: {}", t.name);
            println!("  Dims: {:?}", t.dims);
            println!("  Strides (bytes): {:?}", t.strides);
            println!("  Quant type: {:?}", t.quant_type);

            let ne0 = t.dims.get(0).copied().unwrap_or(0);
            let ne1 = t.dims.get(1).copied().unwrap_or(0);
            let nb0 = t.strides.get(0).copied().unwrap_or(0);
            let nb1 = t.strides.get(1).copied().unwrap_or(0);

            // Block analysis
            let (block_size, block_elements) = match t.quant_type {
                infer_train::types::QuantType::Q5_K => (176, 256),
                infer_train::types::QuantType::Q4_K => (144, 256),
                infer_train::types::QuantType::Q6_K => (210, 256),
                infer_train::types::QuantType::Q4_0 => (18, 32),
                infer_train::types::QuantType::IQ4_XS => (16, 256),
                _ => (nb0 as usize, 1),
            };

            let blocks_per_input_dim =
                (ne1 as usize + block_elements - 1) / block_elements;
            let blocks_per_output_dim =
                (ne0 as usize + block_elements - 1) / block_elements;

            println!(
                "  Block size: {} bytes, {} elements per block",
                block_size, block_elements
            );
            println!(
                "  Blocks needed for input dim (ne1): {}",
                blocks_per_input_dim
            );
            println!(
                "  Blocks needed for output dim (ne0): {}",
                blocks_per_output_dim
            );
            println!(
                "  Expected stride[1] for [ne0][blocks_per_input] = ne0 * block_size = {}",
                ne0 * block_size as u64
            );
            println!(
                "  Expected stride[1] for [blocks_per_input][ne0] = blocks_per_input * block_size = {}",
                (blocks_per_input_dim as u64) * (block_size as u64)
            );
            println!("  Actual stride[1] = {}", nb1);

            // Determine actual layout
            if nb1 == ne0 * block_size as u64 {
                println!(
                    "  Layout: [blocks_per_input][ne0] - Column-major for GEMV"
                );
            } else if nb1 == (blocks_per_input_dim as u64) * (block_size as u64)
            {
                println!(
                    "  Layout: [ne0][blocks_per_input] - Row-major for GEMV"
                );
            } else {
                let interleaving = nb1 / block_size as u64;
                println!(
                    "  Layout: Interleaved with factor {} (stride[1] / block_size)",
                    interleaving
                );
                println!(
                    "  Each input block covers {} output features",
                    interleaving
                );
            }

            break;
        }
    }

    // Create executor once (model already loaded above)
    let mut executor = ForwardExecutor::new_with_threads(
        model,
        cli.threads,
        cli.threads_batch,
    );
    let tokens = executor
        .model
        .tokenizer()
        .encode(&cli.prompt)
        .expect("Failed to tokenize");

    // Warmup
    println!("\n=== Warmup ===");

    for i in 0..cli.warmup {
        println!("Warmup {}/{}...", i + 1, cli.warmup);
        executor.reset(); // Reset KV cache for each warmup
        for &token in &tokens {
            let _ = executor.forward(token);
        }
        for _ in 0..5 {
            if let Ok(logits) = executor.forward(0) {
                let _ = logits;
            }
        }
    }

    // Profile prefill
    println!("\n=== Prefill Profile ===");
    let mut prefill_times = Vec::new();
    println!("Tokens: {} ({:?})", tokens.len(), tokens);

    for _iter in 0..cli.iterations {
        executor.reset(); // Reset KV cache for each iteration

        let start = Instant::now();
        for &token in &tokens {
            let _ = executor.forward(token);
        }
        let elapsed = start.elapsed();
        prefill_times.push(elapsed.as_secs_f64() * 1000.0);
    }

    let avg_prefill_ms: f64 =
        prefill_times.iter().sum::<f64>() / prefill_times.len() as f64;
    println!(
        "Prefill: {:.2} ms ({} tokens, {:.1} t/s)",
        avg_prefill_ms,
        tokens.len(),
        tokens.len() as f64 / (avg_prefill_ms / 1000.0)
    );

    // Profile decode
    println!("\n=== Decode Profile ===");
    let mut decode_times = Vec::new();

    for _iter in 0..cli.iterations {
        executor.reset(); // Reset KV cache for each iteration

        // Prefill first
        for &token in &tokens {
            let _ = executor.forward(token);
        }

        // Decode
        let mut times = Vec::new();
        for _ in 0..cli.decode_tokens {
            let start = Instant::now();
            if let Ok(logits) = executor.forward(0) {
                times.push(start.elapsed().as_secs_f64() * 1000.0);
                // Greedy sampling (not counted)
                let max_idx = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                let _ = max_idx;
            }
        }
        decode_times.extend(times);
    }

    let avg_decode_ms: f64 =
        decode_times.iter().sum::<f64>() / decode_times.len() as f64;
    let min_decode_ms =
        decode_times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_decode_ms =
        decode_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "Decode per token: {:.2} ms avg (min: {:.2}, max: {:.2})",
        avg_decode_ms, min_decode_ms, max_decode_ms
    );
    println!("Decode throughput: {:.1} t/s", 1000.0 / avg_decode_ms);

    // Layer-by-layer analysis (approximation)
    println!("\n=== Per-Layer Approximation ===");
    let ms_per_layer = avg_decode_ms / n_layers as f64;
    println!("~{:.2} ms per layer (total / n_layers)", ms_per_layer);

    // Typical breakdown for Qwen3-0.6B
    // Each layer has: 4 attention GEMV + 3 FFN GEMV + 2 RMS Norm + Attention compute
    // For decode: GEMV dominates (~80% of time)
    let estimated_gemv_time = ms_per_layer * 0.8;
    let n_gemv_per_layer = 7; // q,k,v,o,gate,up,down
    let estimated_gemv_each = estimated_gemv_time / n_gemv_per_layer as f64;

    println!("\nEstimated GEMV breakdown (assuming 80% of layer time):");
    println!("  Total GEMV per layer: {:.2} ms", estimated_gemv_time);
    println!("  Per GEMV (7 per layer): {:.2} ms each", estimated_gemv_each);

    // Memory bandwidth estimation
    println!("\n=== Memory Bandwidth Estimation ===");

    // For Qwen3-0.6B Q4_K:
    // - hidden_size = 896
    // - intermediate_size = 4864
    // - QKV weights: 896 * 896 * 3 = 2.4M elements (Q4_K ~0.5 bytes/element)
    // - O weight: 896 * 896 = 0.8M elements
    // - FFN: 896 * 4864 * 3 = 13M elements

    let hidden_size = config.hidden_size as f64;
    let intermediate_size = config.intermediate_size as f64;

    // Weight read per decode token (bytes)
    // QKV: hidden * hidden * 3 * 0.5 (Q4_K)
    // O: hidden * hidden * 0.5
    // FFN: hidden * intermediate * 3 * 0.5
    let weight_bytes_per_token = (hidden_size * hidden_size * 3.0 * 0.5 + // QKV
         hidden_size * hidden_size * 0.5 +       // O
         hidden_size * intermediate_size * 3.0 * 0.5) * // FFN
        n_layers as f64;

    println!("Weight bytes per token: {:.2} MB", weight_bytes_per_token / 1e6);

    let bandwidth_gbps =
        weight_bytes_per_token / (avg_decode_ms / 1000.0) / 1e9;
    println!("Achieved bandwidth: {:.1} GB/s", bandwidth_gbps);

    // M1 Max theoretical: ~400 GB/s
    // Typically achievable: ~200 GB/s for sequential access
    // For random access: much less
    let theoretical_bw = 400.0;
    let utilization = bandwidth_gbps / theoretical_bw * 100.0;
    println!(
        "Memory bandwidth utilization: {:.1}% (of {} GB/s theoretical)",
        utilization, theoretical_bw
    );

    if utilization < 10.0 {
        println!("⚠ Very low memory bandwidth utilization!");
        println!(
            "  → Likely cause: Random memory access pattern (transposed weights)"
        );
        println!("  → Solution: Weight repacking or interleaved GEMV");
    }

    // Summary
    println!("\n=== Summary ===");
    println!(
        "Decode: {:.1} t/s ({:.2} ms/token)",
        1000.0 / avg_decode_ms,
        avg_decode_ms
    );
    println!(
        "Memory bandwidth: {:.1} GB/s ({:.1}% utilization)",
        bandwidth_gbps, utilization
    );

    // Target comparison
    println!("\n=== Target Comparison ===");
    let target_tps = 200.0; // llama.cpp on M1
    let target_ms = 5.0;
    let gap = avg_decode_ms / target_ms;
    println!("Our decode: {:.1} t/s", 1000.0 / avg_decode_ms);
    println!("Target (llama.cpp): {:.1} t/s", target_tps);
    println!("Gap: {:.1}x slower", gap);

    // Bottleneck identification
    println!("\n=== Bottleneck Analysis ===");
    if utilization < 15.0 {
        println!("PRIMARY BOTTLENECK: Memory access pattern");
        println!("  - Bandwidth utilization only {:.1}%", utilization);
        println!("  - Transposed weight layout causes random access");
        println!(
            "  - Each GEMV reads weights with stride {} KB",
            gguf.tensors
                .iter()
                .find(|t| t.name.contains("attn_q.weight"))
                .map(|t| t.strides.get(1).copied().unwrap_or(0) / 1024)
                .unwrap_or(0)
        );
    }
}
