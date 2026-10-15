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

    // Show key metadata
    println!("\n=== Key Metadata ===");
    let keys = [
        "general.architecture",
        "general.name",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.attention.key_length",
        "llama.attention.value_length",
        "llama.attention.layer_norm_rms_epsilon",
        "llama.block_count",
        "llama.embedding_length",
        "llama.feed_forward_length",
        "qwen35.attention.head_count",
        "qwen35.attention.head_count_kv",
        "qwen35.attention.key_length",
        "qwen35.attention.value_length",
        "qwen35.block_count",
        "qwen35.embedding_length",
        "qwen35.feed_forward_length",
        "qwen35.full_attention_interval",
        "qwen35.rope.dimension_count",
        "mistral3.attention.head_count",
        "mistral3.attention.head_count_kv",
        "mistral3.attention.key_length",
        "mistral3.attention.value_length",
        "mistral3.block_count",
        "mistral3.embedding_length",
        "mistral3.feed_forward_length",
        "mistral3.rope.dimension_count",
    ];
    for key in &keys {
        if let Some(value) = gguf.metadata.get(*key) {
            println!("{}: {:?}", key, value);
        }
    }

    // Print all metadata keys
    println!("\n=== All Metadata Keys ===");
    for key in gguf.metadata.keys() {
        println!("{}", key);
    }

    // Show all tensors that contain "attn_q", "attn_k", or "attn_v"
    println!("\n=== Attention Q/K/V Tensors ===");
    for info in gguf.tensors.iter() {
        if info.name.contains("attn_q")
            || info.name.contains("attn_k")
            || info.name.contains("attn_v")
            || info.name.contains("attn_qkv")
            || info.name.contains("linear_attn")
        {
            println!(
                "{}: shape={:?}, dtype={:?}, offset={}",
                info.name, info.dims, info.dtype, info.offset
            );
        }
    }

    // Show all tensors for layer 0
    println!("\n=== Layer 0 All Tensors ===");
    for info in gguf.tensors.iter() {
        if info.name.starts_with("blk.0.") {
            println!(
                "{}: shape={:?}, dtype={:?}, offset={}",
                info.name, info.dims, info.dtype, info.offset
            );
        }
    }

    // Show all tensors for layer 3 (full attention layer)
    println!("\n=== Layer 3 All Tensors ===");
    for info in gguf.tensors.iter() {
        if info.name.starts_with("blk.3.") {
            println!(
                "{}: shape={:?}, dtype={:?}, offset={}",
                info.name, info.dims, info.dtype, info.offset
            );
        }
    }

    // Show output and embedding tensors
    println!("\n=== Output/Embedding Tensors ===");
    for info in gguf.tensors.iter() {
        if info.name.contains("output")
            || info.name.contains("embd")
            || info.name.contains("lm_head")
        {
            println!(
                "{}: shape={:?}, dtype={:?}, offset={}",
                info.name, info.dims, info.dtype, info.offset
            );
        }
    }
}
