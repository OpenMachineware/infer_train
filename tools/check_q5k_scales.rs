// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

//! Check Q5_K weight scales values

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Model: {}", model_path);

    // Load model
    let ctx = infer_train::inference::InferenceContext::from_gguf(
        model_path,
        infer_train::inference::Device::Cpu,
    )
    .expect("Failed to load model");

    let layer = ctx.model.layers.first().expect("No layers");
    let wq = layer.wq.as_ref().expect("No wq");

    println!("\n=== Q5_K scales check ===");
    println!("wq dtype: {:?}", wq.dtype);
    println!("wq shape: {:?}", wq.shape);

    // Check first few blocks
    let data = wq.data.as_bytes();
    let block_size = std::mem::size_of::<infer_train::quant::types::BlockQ5K>();
    println!("\nBlock size: {} bytes", block_size);
    println!("Total blocks: {}", data.len() / block_size);

    for i in 0..5.min(data.len() / block_size) {
        let offset = i * block_size;
        let block: &infer_train::quant::types::BlockQ5K = unsafe {
            &*(data.as_ptr().add(offset)
                as *const infer_train::quant::types::BlockQ5K)
        };

        println!("\nBlock {}:", i);
        println!("  d: {} (F16 bits)", block.d);
        println!("  d as f16: {:.6}", half::f16::from_bits(block.d).to_f32());
        println!("  dmin: {} (F16 bits)", block.dmin);
        println!(
            "  dmin as f16: {:.6}",
            half::f16::from_bits(block.dmin).to_f32()
        );

        println!("  scales: {:?}", block.scales);

        // Decode scales using the Metal kernel formula
        for j in 0..8 {
            let sc = if j < 4 {
                (block.scales[j + 0] & 63, block.scales[j + 4] & 63)
            } else {
                (
                    (block.scales[j + 4] & 0xF)
                        | ((block.scales[j - 4] & 0xC0) >> 2),
                    (block.scales[j + 4] >> 4)
                        | ((block.scales[j] & 0xC0) >> 2),
                )
            };
            println!("  Sub-block {}: scale={}, min={}", j, sc.0, sc.1);
        }
    }
}
