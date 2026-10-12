// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test model layer loading with all GGUF files
use infer_train::model::unified::{Architecture, Model};

fn main() {
    println!("=== Model Layer Integration Test ===\n");

    let test_files = [
        "Qwen3-0.6B-UD-Q4_K_XL.gguf",
        "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L.gguf",
        "Qwen3.8-27B-UD-Q5_K_M.gguf",
        "Qwen3.6-35B-A3B-DSV4Pro-Distill-MTP-Q5_K_M-imatrix.gguf",
        "Hy-MT2-7B-Q4_K_M.gguf",
        "Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf",
        "Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
    ];

    for file in &test_files {
        let path = format!("/Users/jia/Desktop/infer_train/{}", file);
        match Model::from_gguf(&path) {
            Ok(model) => {
                let arch_str = match model.architecture {
                    Architecture::Llama => "Llama",
                    Architecture::MoE => "MoE",
                    Architecture::Mamba => "Mamba",
                    Architecture::Hybrid => "Hybrid (MoE+SSM)",
                    Architecture::Unknown => "Unknown",
                };
                let complete = model.complete_layers();
                let total = model.layer_count();
                let params = model.param_count();
                let pct = if total > 0 { (complete * 100) / total } else { 0 };
                println!(
                    "{}: {} layers, {}/{} complete ({}%), {} params",
                    file, arch_str, complete, total, pct, params
                );
            }
            Err(e) => {
                println!("{}: Failed to load - {}", file, e);
            }
        }
    }
}
