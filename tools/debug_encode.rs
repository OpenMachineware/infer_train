// Debug encode operations
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

use infer_train::inference::Device;
use infer_train::inference::InferenceContext;
use infer_train::model::tensor::DataType;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Model: {}", model_path);

    let gpu_ctx = InferenceContext::from_gguf(model_path, Device::Gpu)
        .expect("Failed to load GPU model");

    println!("\nChecking layer weight formats:");
    for (i, layer) in gpu_ctx.model.layers.iter().enumerate() {
        println!("\nLayer {}:", i);
        println!("  wq: {:?}", layer.wq.as_ref().map(|t| t.dtype));
        println!("  wk: {:?}", layer.wk.as_ref().map(|t| t.dtype));
        println!("  wv: {:?}", layer.wv.as_ref().map(|t| t.dtype));
        println!("  wo: {:?}", layer.wo.as_ref().map(|t| t.dtype));
        println!("  w1: {:?}", layer.w1.as_ref().map(|t| t.dtype));
        println!("  w2: {:?}", layer.w2.as_ref().map(|t| t.dtype));
        println!("  w3: {:?}", layer.w3.as_ref().map(|t| t.dtype));
    }

    println!("\nOutput: {:?}", gpu_ctx.model.output.as_ref().map(|t| (t.dtype, &t.shape)));

    // Check which formats are supported by mv_encode
    println!("\nSupported formats by mv_encode:");
    let supported = [
        DataType::Q4_0, DataType::Q4_1, DataType::Q5_0, DataType::Q5_1, DataType::Q8_0,
        DataType::Q2_K, DataType::Q3_K, DataType::Q4_K, DataType::Q5_K, DataType::Q6_K,
        DataType::IQ4_NL, DataType::IQ4_XS, DataType::IQ1_S,
        DataType::IQ2_XXS, DataType::IQ2_XS, DataType::IQ2_S,
        DataType::IQ3_XXS, DataType::IQ3_S,
    ];
    for dtype in supported {
        println!("  {:?}", dtype);
    }
}
