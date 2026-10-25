// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Trace NaN in decode operation

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

        println!("\n=== Prefill + Decode with NaN tracing ===");

        // Enable NaN tracing
        // unsafe { env::set_var("INFER_TRACE_NAN", "1"); }

        // Also try with debug tracing
        unsafe {
            env::set_var("INFER_DEBUG_DAG", "1");
        }

        // First do prefill with multiple tokens to build context
        let prefill_tokens: Vec<u32> = (1..=10).collect(); // 10 tokens for prefill
        println!("Prefilling with {} tokens...", prefill_tokens.len());
        let prefill_logits = ctx.forward_gpu_native(&prefill_tokens);
        println!(
            "Prefill logits: {:?}",
            prefill_logits
                .as_ref()
                .map(|l| (l.len(), l.iter().filter(|x| x.is_nan()).count()))
        );

        // Then do decode with single token
        println!("\nDecoding with single token...");
        let decode_token = vec![11u32];
        let decode1 = ctx.forward_gpu_native(&decode_token);
        println!(
            "Decode 1: {:?}",
            decode1
                .as_ref()
                .map(|l| (l.len(), l.iter().filter(|x| x.is_nan()).count()))
        );

        let mut logits = ctx.forward_gpu_native(&decode_token);
        println!(
            "Decode 2: {:?}",
            logits
                .as_ref()
                .map(|l| (l.len(), l.iter().filter(|x| x.is_nan()).count()))
        );

        // For async pipeline, need to call twice to get result
        if logits.is_none() {
            println!("Third decode call...");
            logits = ctx.forward_gpu_native(&decode_token);
            println!(
                "Decode 3: {:?}",
                logits.as_ref().map(|l| (
                    l.len(),
                    l.iter().filter(|x| x.is_nan()).count()
                ))
            );
        }

        match logits {
            Some(l) => {
                let total = l.len();
                let nan_count = l.iter().filter(|x| x.is_nan()).count();
                let inf_count = l.iter().filter(|x| x.is_infinite()).count();
                let zero_count = l.iter().filter(|x| **x == 0.0).count();
                let valid_count =
                    l.iter().filter(|x| x.is_finite() && **x != 0.0).count();

                println!(
                    "Logits: total={}, NaN={}, Inf={}, zero={}, valid={}",
                    total, nan_count, inf_count, zero_count, valid_count
                );

                if valid_count > 0 {
                    let valid_vals: Vec<f32> = l
                        .iter()
                        .filter(|x| x.is_finite() && **x != 0.0)
                        .cloned()
                        .collect();
                    let min = valid_vals
                        .iter()
                        .cloned()
                        .fold(f32::INFINITY, f32::min);
                    let max = valid_vals
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = valid_vals.iter().sum();
                    let avg = sum / valid_vals.len() as f32;
                    println!(
                        "Valid range: min={:.6}, max={:.6}, avg={:.6}",
                        min, max, avg
                    );

                    // Check first 10 non-NaN values
                    println!("\nFirst 10 non-NaN values:");
                    for (i, val) in l.iter().enumerate().take(10) {
                        if val.is_finite() {
                            println!("  logits[{}] = {:.6}", i, val);
                        }
                    }
                }

                if nan_count > 0 || inf_count > 0 {
                    eprintln!(
                        "\n!!! NaN/Inf detected - check TRACE output above !!!"
                    );
                }
            }
            None => {
                eprintln!("Decode failed - no logits returned");
            }
        }

        unsafe {
            env::remove_var("INFER_TRACE_NAN");
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}
