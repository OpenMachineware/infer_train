// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Check for NaN after GPU execution

use std::time::Instant;

fn main() {
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string()
    });

    println!("=== NaN Check After GPU ===\n");

    let result = std::thread::spawn(move || {
        use infer_train::inference::{InferenceContext, KVCacheGPU};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device().clone();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            &model_path,
            gpu_ctx.clone(),
        )
        .unwrap();

        let mut kv = KVCacheGPU::new(&device, &ctx.model.config, 2048);

        // Prefill
        println!("=== Prefill ===");
        let prefill_tokens: Vec<u32> = (1..=64).collect();
        let result = ctx.model.forward_gpu_native_batched(
            &prefill_tokens,
            &mut kv,
            &gpu_ctx,
        );

        if let Some(logits) = &result {
            let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
            let inf_count = logits.iter().filter(|&&x| x.is_infinite()).count();
            let non_zero = logits.iter().filter(|&&x| x != 0.0).count();
            println!(
                "Logits: {} total, {} NaN, {} Inf, {} non-zero",
                logits.len(),
                nan_count,
                inf_count,
                non_zero
            );
            println!("First 10 logits: {:?}", &logits[..10.min(logits.len())]);
        }

        // Decode - 逐个token检查
        println!("\n=== Decode ===");
        for i in 0..10 {
            let token = (65 + i) as u32;
            let start = Instant::now();
            let result = ctx.model.forward_gpu_native_batched(
                &[token],
                &mut kv,
                &gpu_ctx,
            );
            let elapsed = start.elapsed();

            if let Some(logits) = &result {
                let nan_count = logits.iter().filter(|&&x| x.is_nan()).count();
                let inf_count =
                    logits.iter().filter(|&&x| x.is_infinite()).count();
                let non_zero = logits.iter().filter(|&&x| x != 0.0).count();

                println!(
                    "Token {}: {:?} - NaN: {}/{}, Inf: {}/{}, non-zero: {}/{}",
                    i + 1,
                    elapsed,
                    nan_count,
                    logits.len(),
                    inf_count,
                    logits.len(),
                    non_zero,
                    logits.len()
                );

                if nan_count > 0 || inf_count > 0 {
                    println!(
                        "  First 10: {:?}",
                        &logits[..10.min(logits.len())]
                    );
                }
            }
        }
    });

    result.join().unwrap();
}
