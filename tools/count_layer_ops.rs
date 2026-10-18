// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Count operations per layer

use infer_train::inference::{Device, InferenceContext};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    let model = &ctx.model;

    println!("=== Operations per layer ===\n");

    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let mut op_count = 0;

        // 1. RMS norm: 1 kernel
        op_count += 1;

        // 2. QKV projections: 3 kernels
        op_count += 3;

        // 3. RoPE: 2 kernels (Q and K)
        op_count += 2;

        // 4. KV update: 2 kernels
        op_count += 2;

        // 5. Flash attention: 1 kernel
        op_count += 1;

        // 6. Output projection: 1 kernel
        op_count += 1;

        // 7. Residual add: 1 kernel
        op_count += 1;

        // 8. FFN norm: 1 kernel
        op_count += 1;

        // 9. FFN: gate + up + SiLU + mul + down = 5 kernels
        op_count += 5;

        // 10. Final residual add: 1 kernel
        op_count += 1;

        println!(
            "Layer {}: {} operations ({} GEMV, {} element-wise, {} other)",
            layer_idx,
            op_count,
            4, // Q, K, V, output projection
            3, // add, SiLU, mul
            op_count - 7
        );

        if layer_idx >= 3 {
            println!("...");
            break;
        }
    }

    println!("\n=== Total operations ===");
    println!(
        "2 layers: ~{} operations in single command buffer",
        17 * 2
    );
    println!(
        "This likely exceeds Metal's internal limits for a single command buffer"
    );

    println!("\n=== llama.cpp approach ===");
    println!("Split graph into chunks of ~64 operations per command buffer");
    println!("Use 1-2 additional command buffers for the rest");
    println!("Encode and commit in parallel");
}
