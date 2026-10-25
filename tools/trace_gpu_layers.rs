// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Trace GPU inference layer by layer to find where NaN appears

use std::env;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext};
        use infer_train::model::gguf::GgufFile;

        let args: Vec<String> = env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: {} <model.gguf>", args[0]);
            std::process::exit(1);
        }

        let model_path = &args[1];
        println!("Model: {}", model_path);

        // Load GGUF to inspect architecture
        let gguf = GgufFile::parse(model_path).expect("Failed to parse GGUF");
        let arch = gguf.metadata
            .get("general.architecture")
            .and_then(|v| if let infer_train::model::gguf::MetadataValue::String(s) = v {
                Some(s.clone())
            } else {
                None
            })
            .unwrap_or_default();

        println!("Architecture: {}", arch);
        println!("Layers: {}", gguf.metadata
            .get(&format!("{}.block_count", arch))
            .and_then(|v| if let infer_train::model::gguf::MetadataValue::Uint64(n) = v {
                Some(*n)
            } else {
                None
            })
            .unwrap_or(0));

        // Load model for GPU
        let mut ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
            .expect("Failed to load GPU model");

        // Test tokens
        let tokens: Vec<u32> = (1..=5).collect();

        println!("\n=== Tracing GPU layer outputs ===");

        // Run prefill and check each layer
        let result = ctx.forward_gpu_native(&tokens);

        match result {
            Some(logits) => {
                let nan_count = logits.iter().filter(|x| x.is_nan()).count();
                println!("\nFinal logits: total={}, NaN={}", logits.len(), nan_count);

                if nan_count < logits.len() / 2 {
                    let valid: Vec<f32> = logits
                        .iter()
                        .filter(|x| x.is_finite() && **x != 0.0)
                        .cloned()
                        .collect();
                    if !valid.is_empty() {
                        println!(
                            "Valid range: [{:.6}, {:.6}]",
                            valid.iter().cloned().fold(f32::INFINITY, f32::min),
                            valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                        );
                    }
                }
            }
            None => {
                println!("\nGPU inference returned None - trying second call for async pipeline");
                let result2 = ctx.forward_gpu_native(&tokens);
                match result2 {
                    Some(logits) => {
                        let nan_count = logits.iter().filter(|x| x.is_nan()).count();
                        println!("Second call logits: total={}, NaN={}", logits.len(), nan_count);
                    }
                    None => {
                        println!("Second call also returned None");
                    }
                }
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This tool requires Metal support on aarch64");
    }
}
