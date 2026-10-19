// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Check if weights are corrupted at load time (before GPU upload)

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();
        let model = &ctx.model;

        println!(
            "\n=== Checking CPU weight data (before any GPU operations) ==="
        );

        // Check layer 0 weights
        let layer = &model.layers[0];

        // WQ
        if let Some(ref wq) = layer.wq {
            let wq_bytes = wq.bytes();
            let wq_floats = unsafe {
                std::slice::from_raw_parts(
                    wq_bytes.as_ptr() as *const f32,
                    wq_bytes.len() / 4,
                )
            };
            let nan_count = wq_floats.iter().filter(|&&v| v.is_nan()).count();
            let inf_count =
                wq_floats.iter().filter(|&&v| v.is_infinite()).count();
            println!(
                "WQ CPU data: {} NaN, {} inf, {} total floats",
                nan_count,
                inf_count,
                wq_floats.len()
            );
            println!("  First 5: {:?}", &wq_floats[..5.min(wq_floats.len())]);
        }

        // WK
        if let Some(ref wk) = layer.wk {
            let wk_bytes = wk.bytes();
            let wk_floats = unsafe {
                std::slice::from_raw_parts(
                    wk_bytes.as_ptr() as *const f32,
                    wk_bytes.len() / 4,
                )
            };
            let nan_count = wk_floats.iter().filter(|&&v| v.is_nan()).count();
            let inf_count =
                wk_floats.iter().filter(|&&v| v.is_infinite()).count();
            println!(
                "WK CPU data: {} NaN, {} inf, {} total floats",
                nan_count,
                inf_count,
                wk_floats.len()
            );
            println!("  First 5: {:?}", &wk_floats[..5.min(wk_floats.len())]);
        }

        // WV
        if let Some(ref wv) = layer.wv {
            let wv_bytes = wv.bytes();
            let wv_floats = unsafe {
                std::slice::from_raw_parts(
                    wv_bytes.as_ptr() as *const f32,
                    wv_bytes.len() / 4,
                )
            };
            let nan_count = wv_floats.iter().filter(|&&v| v.is_nan()).count();
            let inf_count =
                wv_floats.iter().filter(|&&v| v.is_infinite()).count();
            println!(
                "WV CPU data: {} NaN, {} inf, {} total floats",
                nan_count,
                inf_count,
                wv_floats.len()
            );
            println!("  First 5: {:?}", &wv_floats[..5.min(wv_floats.len())]);
        }

        // Check if it's quantized data (not f32)
        println!("\n=== Weight dtype ===");
        if let Some(ref wq) = layer.wq {
            println!("WQ dtype: {:?}", wq.dtype);
        }
        if let Some(ref wk) = layer.wk {
            println!("WK dtype: {:?}", wk.dtype);
        }
        if let Some(ref wv) = layer.wv {
            println!("WV dtype: {:?}", wv.dtype);
        }

        println!("\n=== Conclusion ===");
        println!(
            "If CPU weights are OK, the corruption happens during GPU upload"
        );
        println!("If CPU weights have NaN, the corruption is in model loading");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
