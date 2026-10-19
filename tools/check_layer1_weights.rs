// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Check Layer 1 weight values for any issues

fn main() {
    let model_path =
        "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext};

        // Load model
        let ctx = InferenceContext::from_gguf(model_path, Device::Gpu).unwrap();

        // Check all layers' WQ first bytes
        for layer_idx in 0..3 {
            let wq = ctx.model.layers[layer_idx].wq.as_ref().unwrap();
            let bytes = wq.bytes();

            // Check first Q5_K block
            // BlockQ5K: d: u16, dmin: u16, scales: [u8; 12], qh: [u8; 32], qs: [u8; 128]
            // Total: 176 bytes

            println!("Layer {} WQ:", layer_idx);
            println!("  First 20 bytes: {:02x?}", &bytes[..20]);

            // Extract d and dmin (stored as half/f16)
            let d_bytes = u16::from_le_bytes([bytes[0], bytes[1]]);
            let dmin_bytes = u16::from_le_bytes([bytes[2], bytes[3]]);
            let d_f16 = half::f16::from_bits(d_bytes);
            let dmin_f16 = half::f16::from_bits(dmin_bytes);

            println!("  d = {} (raw: {:04x})", d_f16, d_bytes);
            println!("  dmin = {} (raw: {:04x})", dmin_f16, dmin_bytes);
            println!("  scales: {:02x?}", &bytes[4..16]);

            // Check for NaN in d or dmin
            if d_f16.is_nan() || dmin_f16.is_nan() {
                println!("  WARNING: NaN in scale values!");
            }

            // Check for zero scale
            if d_f16 == half::f16::ZERO {
                println!("  WARNING: Zero scale!");
            }
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("GPU tests require aarch64+metal");
}
