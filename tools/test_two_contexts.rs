// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test with TWO completely separate contexts

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        // Two COMPLETELY separate contexts
        println!("Creating context 1...");
        let mut ctx1 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

        println!("Creating context 2...");
        let mut ctx2 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

        println!("\nRunning forward pass 1...");
        let r1 = ctx1.model.forward_gpu_native_batched(
            &[1],
            ctx1.gpu_kv_cache.as_mut().unwrap(),
            ctx1.gpu_ctx.as_ref().unwrap(),
        );

        println!("\nRunning forward pass 2...");
        let r2 = ctx2.model.forward_gpu_native_batched(
            &[1],
            ctx2.gpu_kv_cache.as_mut().unwrap(),
            ctx2.gpu_ctx.as_ref().unwrap(),
        );

        if let (Some(r1), Some(r2)) = (r1, r2) {
            let max_diff: f32 = r1
                .iter()
                .zip(r2.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            println!("\nMax diff: {:.6}", max_diff);
            println!("r1 first 5: {:?}", &r1[..5]);
            println!("r2 first 5: {:?}", &r2[..5]);

            if max_diff < 0.001 {
                println!("✓ Deterministic");
            } else {
                println!("✗ NON-DETERMINISTIC");
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
