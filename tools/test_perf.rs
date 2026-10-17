// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test GPU vs CPU performance with more tokens
use infer_train::inference::{InferenceContext, KVCache, KVCacheFormat};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }
    let model_path = &args[1];

    // Test with 10 tokens
    let tokens: Vec<u32> = (1..=10).collect();
    println!("Tokens: {:?}", tokens);

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        // GPU test
        println!("\n=== GPU ===");
        let mut ctx = InferenceContext::from_gguf(
            model_path,
            infer_train::inference::Device::Gpu,
        )
        .expect("Failed to load GPU context");

        let start = std::time::Instant::now();
        let result = ctx.forward_gpu_native(&tokens);
        let gpu_time = start.elapsed();

        println!("GPU time: {:?}", gpu_time);
        if let Some(logits) = result {
            println!(
                "Logits: {} values, max={}",
                logits.len(),
                logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }

        // CPU test
        println!("\n=== CPU ===");
        let mut ctx = InferenceContext::from_gguf(
            model_path,
            infer_train::inference::Device::Cpu,
        )
        .expect("Failed to load CPU context");
        let mut kv = KVCache::new(&ctx.config, 16, KVCacheFormat::F32);

        let start = std::time::Instant::now();
        let result = ctx.model().forward(&tokens, &mut kv);
        let cpu_time = start.elapsed();

        println!("CPU time: {:?}", cpu_time);
        if let Some(logits) = result {
            println!(
                "Logits: {} values, max={}",
                logits.len(),
                logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }

        println!(
            "\nSpeedup: {:.2}x",
            cpu_time.as_secs_f64() / gpu_time.as_secs_f64().max(0.001)
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        // CPU only
        println!("\n=== CPU ===");
        let mut ctx = InferenceContext::from_gguf(
            model_path,
            infer_train::inference::Device::Cpu,
        )
        .expect("Failed to load CPU context");
        let mut kv = KVCache::new(&ctx.config, 16, KVCacheFormat::F32);

        let start = std::time::Instant::now();
        let result = ctx.model().forward(&tokens, &mut kv);
        let cpu_time = start.elapsed();

        println!("CPU time: {:?}", cpu_time);
        if let Some(logits) = result {
            println!(
                "Logits: {} values, max={}",
                logits.len(),
                logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }
    }
}
