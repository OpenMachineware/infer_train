// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Stability test for GPU inference: Run 1000+ tokens and monitor for issues

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf> [num_tokens]", args[0]);
        eprintln!("  num_tokens: Number of tokens to generate (default: 1000)");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let num_tokens: usize =
        args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);

    println!("=== GPU Stability Test ===");
    println!("Model: {}", model_path);
    println!("Target tokens: {}", num_tokens);
    println!();

    // Load model on GPU
    let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model on GPU");

    println!("✓ Model loaded");
    println!("  Layers: {}", ctx.layer_count());
    println!("  Params: {}M", ctx.param_count() / 1_000_000);
    println!(
        "  Config: vocab={}, hidden={}",
        ctx.config.vocab_size, ctx.config.hidden_size
    );

    // Check output tensor transpose
    if let Some(output) = ctx.model().output.as_ref() {
        println!(
            "  Output tensor: dtype={:?}, shape={:?}",
            output.dtype, output.shape
        );
        if output.shape.len() == 2 && output.shape[0] == ctx.config.vocab_size {
            println!("  ✓ Output layer transposed successfully");
        }
    }
    println!();

    // Run single large forward pass
    println!("Running forward pass with {} tokens...", num_tokens);

    let mut total_time = std::time::Duration::ZERO;
    let mut error_count = 0;
    let mut nan_count = 0;
    let mut inf_count = 0;

    // Generate sequential tokens (just for testing, not meaningful)
    let tokens: Vec<u32> = (1..=num_tokens as u32).collect();

    let start = std::time::Instant::now();

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    let result = ctx.forward_gpu_native(&tokens);

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    let result = {
        let mut kv_cache = KVCache::new(&ctx.config, 2048, KVCacheFormat::F32);
        ctx.forward(&tokens, &mut kv_cache)
    };

    let elapsed = start.elapsed();
    total_time = elapsed;

    match result {
        Some(logits) => {
            nan_count = logits.iter().filter(|&&v| v.is_nan()).count();
            inf_count = logits.iter().filter(|&&v| v.is_infinite()).count();

            println!("✓ Forward pass completed");
            println!("  Time: {:?}", elapsed);
            println!(
                "  Tokens/s: {:.2}",
                num_tokens as f64 / elapsed.as_secs_f64()
            );
            println!("  Logits: {}", logits.len());
            println!("  NaN count: {}", nan_count);
            println!("  Inf count: {}", inf_count);
        }
        None => {
            println!("❌ Forward pass failed");
            error_count = 1;
        }
    }

    let total_elapsed = elapsed;

    println!("\n=== Results ===");
    println!("Total tokens: {}", num_tokens);
    println!("Total time: {:?}", total_elapsed);
    println!(
        "Tokens/s: {:.2}",
        num_tokens as f64 / total_elapsed.as_secs_f64()
    );
    println!("NaN count: {}", nan_count);
    println!("Inf count: {}", inf_count);
    println!("Errors: {}", error_count);

    if error_count == 0 && nan_count == 0 && inf_count == 0 {
        println!("\n✓ Stability test PASSED");
    } else {
        println!("\n✗ Stability test FAILED");
        if error_count > 0 {
            println!("  - Forward pass failed");
        }
        if nan_count > 0 {
            println!("  - {} NaN values detected", nan_count);
        }
        if inf_count > 0 {
            println!("  - {} Inf values detected", inf_count);
        }
    }
}
