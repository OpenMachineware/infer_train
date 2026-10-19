// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug forward_gpu_native_batched step by step

use infer_train::inference::{Device, InferenceContext};
use std::time::Instant;

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        // Test 1: Single token prefill
        println!("\n=== Test 1: Prefill (token=1, fresh context) ===");
        let mut ctx1 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let result1 = ctx1.model.forward_gpu_native_batched(
            &[1],
            ctx1.gpu_kv_cache.as_mut().unwrap(),
            ctx1.gpu_ctx.as_ref().unwrap(),
        );
        if let Some(ref r) = result1 {
            let nan_count = r.iter().filter(|&&v| v.is_nan()).count();
            println!("Result: {} logits, {} NaN", r.len(), nan_count);
        }

        // Test 2: Two separate contexts
        println!("\n=== Test 2: Two separate contexts ===");
        let mut ctx2a =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let result2a = ctx2a.model.forward_gpu_native_batched(
            &[1],
            ctx2a.gpu_kv_cache.as_mut().unwrap(),
            ctx2a.gpu_ctx.as_ref().unwrap(),
        );

        let mut ctx2b =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let result2b = ctx2b.model.forward_gpu_native_batched(
            &[2],
            ctx2b.gpu_kv_cache.as_mut().unwrap(),
            ctx2b.gpu_ctx.as_ref().unwrap(),
        );

        if let (Some(ra), Some(rb)) = (&result2a, &result2b) {
            let nan_a = ra.iter().filter(|&&v| v.is_nan()).count();
            let nan_b = rb.iter().filter(|&&v| v.is_nan()).count();
            println!("Context A (token 1): {} NaN", nan_a);
            println!("Context B (token 2): {} NaN", nan_b);
        }

        // Test 3: Same context, two tokens sequentially
        println!("\n=== Test 3: Same context, sequential tokens ===");
        let mut ctx3 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

        // Token 1
        let result3a = ctx3.model.forward_gpu_native_batched(
            &[1],
            ctx3.gpu_kv_cache.as_mut().unwrap(),
            ctx3.gpu_ctx.as_ref().unwrap(),
        );
        if let Some(ref r) = result3a {
            let nan_count = r.iter().filter(|&&v| v.is_nan()).count();
            println!("Token 1: {} NaN", nan_count);
            println!(
                "KV cache position: {}",
                ctx3.gpu_kv_cache.as_ref().unwrap().seq_len()
            );
        }

        // Token 2
        let result3b = ctx3.model.forward_gpu_native_batched(
            &[2],
            ctx3.gpu_kv_cache.as_mut().unwrap(),
            ctx3.gpu_ctx.as_ref().unwrap(),
        );
        if let Some(ref r) = result3b {
            let nan_count = r.iter().filter(|&&v| v.is_nan()).count();
            println!("Token 2: {} NaN", nan_count);
            println!(
                "KV cache position: {}",
                ctx3.gpu_kv_cache.as_ref().unwrap().seq_len()
            );
        }

        // Test 4: Same context, multiple tokens in single call
        println!("\n=== Test 4: Same context, multiple tokens in one call ===");
        let mut ctx4 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let result4 = ctx4.model.forward_gpu_native_batched(
            &[1, 2],
            ctx4.gpu_kv_cache.as_mut().unwrap(),
            ctx4.gpu_ctx.as_ref().unwrap(),
        );
        if let Some(ref r) = result4 {
            let nan_count = r.iter().filter(|&&v| v.is_nan()).count();
            println!("Result: {} NaN", nan_count);
            println!(
                "KV cache position: {}",
                ctx4.gpu_kv_cache.as_ref().unwrap().seq_len()
            );
        }

        println!("\n=== Summary ===");
        println!(
            "If Test 3 token 2 has NaN but Test 2b doesn't, the issue is in context state"
        );
        println!(
            "If Test 4 has NaN, the issue is in multi-token batch processing"
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
