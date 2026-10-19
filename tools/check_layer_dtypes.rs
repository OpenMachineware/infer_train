// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Check weight dtypes for each layer

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext};

        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

        for layer_idx in 0..ctx.model.layers.len() {
            let layer = &ctx.model.layers[layer_idx];

            let wq_dtype = layer.wq.as_ref().map(|w| w.dtype);
            let wk_dtype = layer.wk.as_ref().map(|w| w.dtype);
            let wv_dtype = layer.wv.as_ref().map(|w| w.dtype);
            let wo_dtype = layer.wo.as_ref().map(|w| w.dtype);
            let w1_dtype = layer.w1.as_ref().map(|w| w.dtype);
            let w2_dtype = layer.w2.as_ref().map(|w| w.dtype);
            let w3_dtype = layer.w3.as_ref().map(|w| w.dtype);

            println!("Layer {}:", layer_idx);
            println!(
                "  wq: {:?}, wk: {:?}, wv: {:?}, wo: {:?}",
                wq_dtype, wk_dtype, wv_dtype, wo_dtype
            );
            println!(
                "  w1: {:?}, w2: {:?}, w3: {:?}",
                w1_dtype, w2_dtype, w3_dtype
            );

            // Check if Layer 1 has any special weights
            if layer_idx == 1 {
                let wq = layer.wq.as_ref().unwrap();
                println!("  Layer 1 WQ shape: {:?}", wq.shape);
                println!("  Layer 1 WQ bytes len: {}", wq.bytes().len());
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
