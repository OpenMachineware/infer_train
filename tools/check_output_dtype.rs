// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Check output tensor dtype

fn main() {
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or("DeepSeek-R1-Distill-Qwen-1.5B-Q5_K_M.gguf".to_string());

    let ctx = infer_train::inference::InferenceContext::from_gguf(
        &model_path,
        infer_train::inference::Device::Cpu,
    )
    .unwrap();

    println!("Config vocab_size: {}", ctx.config.vocab_size);
    println!("Config hidden_size: {}", ctx.config.hidden_size);

    if let Some(output) = &ctx.model.output {
        println!("Output tensor:");
        println!("  dtype: {:?}", output.dtype);
        println!("  shape: {:?}", output.shape);
    } else {
        println!("No output tensor found!");
    }
}
