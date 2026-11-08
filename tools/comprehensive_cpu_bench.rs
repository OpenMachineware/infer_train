// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu
// Comprehensive CPU performance comparison with llama.cpp

use infer_train::cpu::ThreadConfig;
use infer_train::inference::InferenceContext;
use infer_train::model::LlamaModel;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = &args[1];

    println!("=== Comprehensive CPU Benchmark ===");
    println!("Model: {}", model_path);
    println!();

    // Load model
    let model = LlamaModel::from_gguf_path(model_path, 1).expect("Failed to load model");
    println!("Model loaded: {} layers, {} hidden", model.config.n_layers, model.config.hidden_size);
    println!();

    // Test configurations
    let thread_configs = vec![
        ("1T", ThreadConfig { n_threads_decode: 1, n_threads_prefill: 1 }),
        ("4T", ThreadConfig { n_threads_decode: 4, n_threads_prefill: 4 }),
        ("8T", ThreadConfig { n_threads_decode: 8, n_threads_prefill: 8 }),
    ];

    let prefill_lengths = vec![1, 32, 128, 512];

    println!("| Threads | Test | t/s | Time (ms) |");
    println!("|---------|------|-----|-----------|");

    for (thread_name, thread_config) in &thread_configs {
        // Decode tests (single token generation)
        for _ in 0..3 { // Warmup
            let mut ctx = InferenceContext::new_with_config(
                &model,
                thread_config.clone(),
                Default::default(),
            ).unwrap();
            let _ = ctx.forward_single_token_cpu(&[1]);
        }

        let mut ctx = InferenceContext::new_with_config(
            &model,
            thread_config.clone(),
            Default::default(),
        ).unwrap();

        // Measure decode performance
        let start = Instant::now();
        let n_decode = 32;
        for _ in 0..n_decode {
            let _ = ctx.forward_single_token_cpu(&[1]);
        }
        let decode_time = start.elapsed().as_millis();
        let decode_tps = n_decode as f64 / (decode_time as f64 / 1000.0);

        println!("| {} | tg32 | {:.2} | {:.1} |", thread_name, decode_tps, decode_time);

        // Prefill tests
        for &prefill_len in &prefill_lengths {
            let mut ctx = InferenceContext::new_with_config(
                &model,
                thread_config.clone(),
                Default::default(),
            ).unwrap();

            let tokens: Vec<u32> = (0..prefill_len as u32).collect();

            let start = Instant::now();
            let _ = ctx.forward_single_token_cpu(&tokens);
            let prefill_time = start.elapsed().as_millis();
            let prefill_tps = prefill_len as f64 / (prefill_time as f64 / 1000.0);

            println!("| {} | pp{} | {:.2} | {:.1} |",
                thread_name, prefill_len, prefill_tps, prefill_time);
        }
        println!("|---------|------|-----|-----------|");
    }
}
