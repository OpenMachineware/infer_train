// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Compare weights between two model loads

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext};

        // Load model
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

        // Get Layer 1 WQ (Q5_K format)
        let wq = ctx.model.layers[1].wq.as_ref().unwrap();
        let wq_bytes = wq.bytes().to_vec(); // Copy to owned

        println!("WQ shape: {:?}", wq.shape);
        println!("WQ dtype: {:?}", wq.dtype);

        // Check first few bytes
        println!("WQ first 10 bytes: {:?}", &wq_bytes[..10]);
        println!("WQ bytes len: {}", wq_bytes.len());

        // Drop context
        drop(ctx);

        // Load model again
        println!("\n=== Run 2 (new load) ===");
        let ctx2 =
            InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

        // Get Layer 1 WQ again
        let wq2 = ctx2.model.layers[1].wq.as_ref().unwrap();
        let wq2_bytes = wq2.bytes().to_vec(); // Copy to owned

        println!("WQ first 10 bytes: {:?}", &wq2_bytes[..10]);
        println!("WQ bytes len: {}", wq2_bytes.len());

        // Compare
        let diff: usize = wq_bytes
            .iter()
            .zip(wq2_bytes.iter())
            .filter(|(a, b)| a != b)
            .count();
        println!("Bytes differ: {}", diff);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
