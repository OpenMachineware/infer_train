// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test if old implementation is deterministic

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        // Two FRESH contexts with IDENTICAL initial state
        let mut ctx1 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let mut ctx2 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

        let tokens = vec![1u32];

        // Run same implementation twice
        let r1 = ctx1
            .model
            .forward_gpu_native_batched(
                &tokens,
                ctx1.gpu_kv_cache.as_mut().unwrap(),
                ctx1.gpu_ctx.as_ref().unwrap(),
            )
            .unwrap();

        let r2 = ctx2
            .model
            .forward_gpu_native_batched(
                &tokens,
                ctx2.gpu_kv_cache.as_mut().unwrap(),
                ctx2.gpu_ctx.as_ref().unwrap(),
            )
            .unwrap();

        let max_diff = r1
            .iter()
            .zip(r2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("Max diff between two runs of SAME impl: {:.6}", max_diff);
        println!("Run1 first 5: {:?}", &r1[..5]);
        println!("Run2 first 5: {:?}", &r2[..5]);

        if max_diff < 0.001 {
            println!("✓ Deterministic");
        } else {
            println!("✗ NON-DETERMINISTIC");
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
