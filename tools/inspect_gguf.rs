// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// GGUF tensor inspection tool

use infer_train::model::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Opening GGUF file: {}", model_path);

    let gguf = match GgufFile::open(model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error opening GGUF: {}", e);
            std::process::exit(1);
        }
    };

    println!("\n=== File Info ===");
    println!("Version: {}", gguf.version);
    println!("Tensor count: {}", gguf.tensor_count);
    println!("Data offset: {}", gguf.data_offset);

    // Show all tensors that contain "attn_q" or "attn_k"
    println!("\n=== Attention Q/K Tensors ===");
    for info in gguf.tensors.iter() {
        if info.name.contains("attn_q") || info.name.contains("attn_k") {
            println!(
                "{}: shape={:?}, dtype={:?}, offset={}",
                info.name, info.dims, info.dtype, info.offset
            );
        }
    }
}
