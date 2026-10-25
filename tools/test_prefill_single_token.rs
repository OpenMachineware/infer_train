// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Test prefill with 1 token to compare with decode

use std::env;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext};

        let args: Vec<String> = env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: {} <model.gguf>", args[0]);
            std::process::exit(1);
        }

        let model_path = &args[1];
        println!("Model: {}", model_path);

        let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load GPU model");

        // Test with different token counts
        println!("\n=== Testing prefill with 1 token ===");
        let tokens1 = vec![1u32];
        let result1 = ctx.forward_gpu_native(&tokens1);
        println!(
            "1 token prefill: {:?}",
            result1
                .as_ref()
                .map(|l| (l.len(), l.iter().filter(|x| x.is_nan()).count()))
        );

        // Second call to get result (async pipeline)
        if result1.is_none() {
            println!("Second call...");
            let result1b = ctx.forward_gpu_native(&tokens1);
            println!(
                "1 token prefill (2nd call): {:?}",
                result1b.as_ref().map(|l| (
                    l.len(),
                    l.iter().filter(|x| x.is_nan()).count()
                ))
            );
        }

        // Reset context
        let mut ctx2 = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load GPU model");

        println!("\n=== Testing prefill with 2 tokens ===");
        let tokens2 = vec![1u32, 2u32];
        let result2 = ctx2.forward_gpu_native(&tokens2);
        println!(
            "2 tokens prefill: {:?}",
            result2
                .as_ref()
                .map(|l| (l.len(), l.iter().filter(|x| x.is_nan()).count()))
        );

        if result2.is_none() {
            println!("Second call...");
            let result2b = ctx2.forward_gpu_native(&tokens2);
            println!(
                "2 tokens prefill (2nd call): {:?}",
                result2b.as_ref().map(|l| (
                    l.len(),
                    l.iter().filter(|x| x.is_nan()).count()
                ))
            );
        }

        // Reset context
        let mut ctx3 = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load GPU model");

        println!("\n=== Testing prefill with 5 tokens ===");
        let tokens5 = vec![1u32, 2u32, 3u32, 4u32, 5u32];
        let result5 = ctx3.forward_gpu_native(&tokens5);
        println!(
            "5 tokens prefill: {:?}",
            result5
                .as_ref()
                .map(|l| (l.len(), l.iter().filter(|x| x.is_nan()).count()))
        );
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}
