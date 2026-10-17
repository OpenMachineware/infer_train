// Profile GPU performance at different sequence lengths
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::inference::{
    Device, InferenceContext, KVCache, KVCacheFormat,
};

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";

    println!("=== GPU Performance Profile (Batched Prefill) ===\n");

    // Load GPU model
    let mut gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model on GPU");

    println!(
        "Config: hidden={}, layers={}, heads={}\n",
        gpu_ctx.config.hidden_size,
        gpu_ctx.config.n_layers,
        gpu_ctx.config.n_heads
    );

    // Test different sequence lengths
    let test_lengths = vec![1, 8, 32, 128];

    for &len in &test_lengths {
        let tokens: Vec<u32> = (1..=len as u32).collect();

        let start = std::time::Instant::now();

        #[cfg(all(target_arch = "aarch64", feature = "metal"))]
        let logits =
            gpu_ctx.forward_prefill_batch(&tokens).expect("Forward failed");

        #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
        {
            let mut kv_cache =
                KVCache::new(&gpu_ctx.config, 1024, KVCacheFormat::F32);
            let logits = gpu_ctx
                .forward(&tokens, &mut kv_cache)
                .expect("Forward failed");
        }

        let elapsed = start.elapsed();

        let ms_per_token = elapsed.as_millis() as f64 / len as f64;
        let tokens_per_sec = 1000.0 / ms_per_token;

        #[cfg(all(target_arch = "aarch64", feature = "metal"))]
        println!(
            "Len {:4}: {:6.1}ms total, {:.2}ms/token, {:.1} tok/s, logits[0]={:.3}",
            len,
            elapsed.as_millis(),
            ms_per_token,
            tokens_per_sec,
            logits[0]
        );

        #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
        println!(
            "Len {:4}: {:6.1}ms total, {:.2}ms/token, {:.1} tok/s",
            len,
            elapsed.as_millis(),
            ms_per_token,
            tokens_per_sec
        );
    }

    println!("\n=== CPU Comparison ===\n");

    // Load CPU model
    let cpu_ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model on CPU");

    for len in test_lengths.iter().take(3).copied() {
        let tokens: Vec<u32> = (1..=len as u32).collect();

        let start = std::time::Instant::now();
        let mut kv_cache =
            KVCache::new(&cpu_ctx.config, 1024, KVCacheFormat::F32);
        let _logits = cpu_ctx
            .model
            .forward(&tokens, &mut kv_cache)
            .expect("Forward failed");
        let elapsed = start.elapsed();

        let ms_per_token = elapsed.as_millis() as f64 / len as f64;
        let tokens_per_sec = 1000.0 / ms_per_token;

        println!(
            "Len {:4}: {:6.1}ms total, {:.2}ms/token, {:.1} tok/s",
            len,
            elapsed.as_millis(),
            ms_per_token,
            tokens_per_sec
        );
    }
}
