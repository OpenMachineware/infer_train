// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Check each layer's weight types

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext};

        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

        println!("=== Layer Weight Types ===");
        for (i, layer) in ctx.model.layers.iter().enumerate() {
            println!("\nLayer {}:", i);
            if let Some(wq) = &layer.wq {
                println!("  wq: dtype={:?}, shape={:?}", wq.dtype, wq.shape);
            }
            if let Some(wk) = &layer.wk {
                println!("  wk: dtype={:?}, shape={:?}", wk.dtype, wk.shape);
            }
            if let Some(wv) = &layer.wv {
                println!("  wv: dtype={:?}, shape={:?}", wv.dtype, wv.shape);
            }
            if let Some(wo) = &layer.wo {
                println!("  wo: dtype={:?}, shape={:?}", wo.dtype, wo.shape);
            }
            if let Some(w1) = &layer.w1 {
                println!(
                    "  w1 (gate): dtype={:?}, shape={:?}",
                    w1.dtype, w1.shape
                );
            }
            if let Some(w2) = &layer.w2 {
                println!(
                    "  w2 (down): dtype={:?}, shape={:?}",
                    w2.dtype, w2.shape
                );
            }
            if let Some(w3) = &layer.w3 {
                println!(
                    "  w3 (up): dtype={:?}, shape={:?}",
                    w3.dtype, w3.shape
                );
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
