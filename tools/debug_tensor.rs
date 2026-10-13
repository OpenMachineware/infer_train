// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Debug tool to read raw tensor data from GGUF

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

    // Find token_embd.weight tensor
    let emb_tensor = gguf
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .expect("token_embd.weight not found");

    println!("Found token_embd.weight:");
    println!("  Shape: {:?}", emb_tensor.dims);
    println!("  Dtype: {:?}", emb_tensor.dtype);
    println!("  Offset: {}", emb_tensor.offset);

    // Get tensor data
    let data = gguf
        .get_tensor_data("token_embd.weight")
        .expect("Failed to get tensor data");
    println!("  Data length: {} bytes", data.len());

    // Print first 50 bytes
    println!("\n  First 50 bytes:");
    for i in 0..50.min(data.len()) {
        print!("{:02x} ", data[i]);
        if (i + 1) % 16 == 0 {
            println!();
        }
    }
    println!();

    // Calculate expected size
    let elem_count: usize = emb_tensor.dims.iter().product::<u64>() as usize;
    let n_blocks = elem_count / 256;
    let expected_bytes = n_blocks * 176;
    println!(
        "\n  Expected size: {} bytes ({} blocks * 176)",
        expected_bytes, n_blocks
    );
    println!("  Actual size: {} bytes", data.len());

    // Check if the tensor offset is within file bounds
    let file_size = std::fs::metadata(model_path).unwrap().len();
    println!("  File size: {} bytes", file_size);
    println!("  Data start offset: {}", gguf.data_offset + emb_tensor.offset);
    println!(
        "  Data end offset: {}",
        gguf.data_offset + emb_tensor.offset + data.len() as u64
    );
}
