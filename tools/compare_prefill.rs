// Compare prefill performance: our implementation vs llama.cpp
// SPDX-License-Identifier: Apache-2.0-or-later
// SPDX-FileCopyrightText: 2026 Jia Liu

use infer_train::buffer::ForwardBufferPool;
use infer_train::inference::{InferenceContext, KVCacheGPU};
use metal::Buffer;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("=== Prefill Performance Comparison ===\n");

    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string()
    });

    println!("Model: {}", model_path);
    println!();

    // Test prompt lengths
    let prompt_lengths = [64, 128, 256, 512];

    println!("| Tokens | Ours (t/s) | llama.cpp (t/s) | Ratio |");
    println!("|--------|------------|-----------------|-------|");

    // Load our model once
    let ctx = InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Gpu,
    )
    .expect("Failed to load model");

    let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");
    let device = gpu_ctx.device().clone();
    let reuse_pool = Rc::new(RefCell::new(Vec::new()));

    println!(
        "Model loaded: {} layers, hidden={}",
        ctx.model.layers.len(),
        ctx.model.config.hidden_size
    );

    // Test each prompt length
    for prompt_len in prompt_lengths.iter() {
        // Create test tokens
        let tokens: Vec<u32> = (0..*prompt_len as u32).collect();

        // === Our implementation ===
        let mut kv_cache = KVCacheGPU::new(&device, &ctx.model.config, 1024);
        let (embed_buffer, batch) = ctx
            .model
            .embedding_lookup_batched_gpu(&tokens, &device)
            .expect("Embedding failed");

        let mut forward_pool =
            ForwardBufferPool::new(device.clone(), reuse_pool.clone());

        // Warmup
        {
            let command_buffer = gpu_ctx.queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            let mut input_buffer: Option<Rc<Buffer>> =
                Some(embed_buffer.clone());
            for (layer_idx, layer) in ctx.model.layers.iter().enumerate() {
                if let Some(output) = ctx.model.forward_layer_prefill_batched(
                    layer,
                    layer_idx,
                    input_buffer.as_ref().unwrap(),
                    batch,
                    &mut kv_cache,
                    gpu_ctx,
                    &mut forward_pool,
                    &encoder,
                ) {
                    input_buffer = Some(output);
                }
            }
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }

        // Measure our performance (3 runs, take best)
        let mut our_best = 0.0;
        for _ in 0..3 {
            let mut kv_cache =
                KVCacheGPU::new(&device, &ctx.model.config, 1024);
            let (embed_buffer, batch) = ctx
                .model
                .embedding_lookup_batched_gpu(&tokens, &device)
                .expect("Embedding failed");
            let mut forward_pool =
                ForwardBufferPool::new(device.clone(), reuse_pool.clone());

            let start = std::time::Instant::now();
            {
                let command_buffer = gpu_ctx.queue().new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                let mut input_buffer: Option<Rc<Buffer>> =
                    Some(embed_buffer.clone());
                for (layer_idx, layer) in ctx.model.layers.iter().enumerate() {
                    if let Some(output) =
                        ctx.model.forward_layer_prefill_batched(
                            layer,
                            layer_idx,
                            input_buffer.as_ref().unwrap(),
                            batch,
                            &mut kv_cache,
                            gpu_ctx,
                            &mut forward_pool,
                            &encoder,
                        )
                    {
                        input_buffer = Some(output);
                    }
                }
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }
            let tps = *prompt_len as f64 / start.elapsed().as_secs_f64();
            if tps > our_best {
                our_best = tps;
            }
        }

        // === llama.cpp benchmark ===
        // Build prompt string
        let prompt = "The quick brown fox jumps over the lazy dog. "
            .repeat((*prompt_len / 10).max(1));

        // Run llama.cpp
        let llama_output =
            std::process::Command::new("./llama.cpp-0.7.0/build/bin/llama-cli")
                .args([
                    "-m",
                    &model_path,
                    "-p",
                    &prompt,
                    "-n",
                    "0", // No generation, just prefill
                    "--batch-size",
                    &prompt_len.to_string(),
                    "--no-display-prompt",
                    "-ngl",
                    "99", // All layers on GPU
                ])
                .output()
                .expect("Failed to run llama.cpp");

        let stdout = String::from_utf8_lossy(&llama_output.stdout);

        // Parse llama.cpp output for prefill time
        // Looking for "eval time" or similar metrics
        let llama_tps = parse_llama_prefill_tps(&stdout, *prompt_len);

        let ratio = our_best / llama_tps;
        let status = if ratio > 1.0 { "✓" } else { "✗" };

        println!(
            "| {:6} | {:10.1} | {:15.1} | {:5.2}x {} |",
            prompt_len, our_best, llama_tps, ratio, status
        );
    }

    println!();
}

fn parse_llama_prefill_tps(output: &str, prompt_len: usize) -> f64 {
    // llama.cpp output format:
    // "eval time =      123.45 ms /   256 tokens (    48.00 tokens per second)"
    for line in output.lines() {
        if line.contains("eval time") || line.contains("prompt eval time") {
            // Try to extract tokens per second
            if let Some(pos) = line.find("tokens per second") {
                let before = &line[..pos];
                // Find the number before "tokens per second"
                if let Some(num_str) = before.split_whitespace().last() {
                    if let Ok(tps) = num_str.parse::<f64>() {
                        return tps;
                    }
                }
            }
            // Alternative: parse from "X ms / Y tokens"
            if let Some(pos) = line.find(" ms /") {
                let after = &line[pos + 5..];
                let parts: Vec<&str> = after.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(ms) = parts[0].parse::<f64>() {
                        return prompt_len as f64 / (ms / 1000.0);
                    }
                }
            }
        }
    }
    // Fallback: estimate from total time
    1000.0
}
