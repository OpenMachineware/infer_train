// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Check: Layer weight types and shapes

use infer_train::inference::{Device, InferenceContext};
use infer_train::model::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];

    let gguf = GgufFile::open(model_path).expect("Failed to open GGUF");
    let ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load model");

    let model = &ctx.model;

    println!("\n=== Layer Weight Check ===\n");

    for layer_idx in [0, 1, 2, 3, 4].iter() {
        let layer = &model.layers[*layer_idx];

        println!("Layer {}:", layer_idx);

        // Q weight
        if let Some(ref wq) = layer.wq {
            let data = wq.data.as_bytes();
            println!(
                "  WQ: shape={:?}, dtype={:?}, n_bytes={}",
                wq.shape,
                wq.dtype,
                data.len()
            );
            // Check first few bytes
            if data.len() >= 32 {
                let first_bytes = &data[0..32.min(data.len())];
                println!("    First 32 bytes: {:?}", first_bytes);
            }
        }

        // K weight
        if let Some(ref wk) = layer.wk {
            let data = wk.data.as_bytes();
            println!(
                "  WK: shape={:?}, dtype={:?}, n_bytes={}",
                wk.shape,
                wk.dtype,
                data.len()
            );
            if data.len() >= 32 {
                let first_bytes = &data[0..32.min(data.len())];
                println!("    First 32 bytes: {:?}", first_bytes);
            }
        }

        // V weight
        if let Some(ref wv) = layer.wv {
            let data = wv.data.as_bytes();
            println!(
                "  WV: shape={:?}, dtype={:?}, n_bytes={}",
                wv.shape,
                wv.dtype,
                data.len()
            );
        }

        println!();
    }
}
