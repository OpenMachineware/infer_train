// Debug script to analyze GGUF weight layout
// Run with: cargo run --bin debug_layout

use infer_train::model::GgufFile;

fn get_block_info(qt: infer_train::types::QuantType) -> (usize, usize) {
    use infer_train::types::QuantType::*;
    match qt {
        F32 => (1, 4),
        F16 => (1, 2),
        Q4_0 => (32, 18), // 32 elements, 18 bytes per block
        Q4_1 => (32, 20),
        Q5_0 => (32, 22),
        Q5_1 => (32, 24),
        Q8_0 => (32, 34),
        Q8_1 => (32, 36),
        Q2_K => (256, 66),
        Q3_K => (256, 86),
        Q4_K => (256, 144),
        Q5_K => (256, 176),
        Q6_K => (256, 210),
        Q8_K => (256, 322),
        IQ4_XS => (256, 138),
        _ => (1, 4), // default
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    // Key weight patterns
    let key_patterns = [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output.weight",
        "token_embd.weight",
    ];

    println!("=== GGUF Weight Layout Analysis ===\n");
    println!(
        "Format: [ne[0], ne[1]] where ne[0] is inner dimension (columns)\n"
    );

    for pattern in &key_patterns {
        if let Some(tensor) = gguf.tensors.iter().find(|t| t.name == *pattern) {
            println!("{}:", pattern);
            println!("  dims: {:?}", tensor.dims);
            println!("  quant_type: {:?}", tensor.quant_type);

            let (blck_size, blck_bytes) = get_block_info(tensor.quant_type);
            println!("  block_size: {} elements", blck_size);
            println!("  block_bytes: {} bytes", blck_bytes);

            if tensor.dims.len() == 2 {
                let ne0 = tensor.dims[0] as usize; // columns (inner dim)
                let ne1 = tensor.dims[1] as usize; // rows (outer dim)
                let n_blocks_per_row = ne0 / blck_size;
                let total_blocks = ne0 * ne1 / blck_size;

                println!("  ne[0] (inner dim): {}", ne0);
                println!("  ne[1] (outer dim): {}", ne1);
                println!("  blocks per row: {}", n_blocks_per_row);
                println!("  total blocks: {}", total_blocks);

                // Verify: for linear weight W[out_features, in_features]
                // GGUF stores as [in_features, out_features] = [ne0, ne1]
                // So: ne0 = in_features, ne1 = out_features
                // For Q projection: input = hidden_size, output = n_heads * head_dim
                // For FFN gate: input = hidden_size, output = intermediate_size

                println!("  Layout interpretation (row-major):");
                println!(
                    "    blocks[row][col] = blocks[row * n_blocks_per_row + col]"
                );
                println!("    row 0: blocks 0..{}", n_blocks_per_row);
                println!(
                    "    row 1: blocks {}..{}",
                    n_blocks_per_row,
                    2 * n_blocks_per_row
                );
            }
            println!();
        }
    }

    // Print model config
    println!("=== Model Config ===");
    if let Some(arch) = gguf.kv.get("general.architecture") {
        println!("architecture: {:?}", arch);
    }
    if let Some(hs) = gguf.kv.get("llama.embedding_length") {
        println!("hidden_size: {:?}", hs);
    }
    if let Some(nl) = gguf.kv.get("llama.block_count") {
        println!("n_layers: {:?}", nl);
    }
    if let Some(nh) = gguf.kv.get("llama.attention.head_count") {
        println!("n_heads: {:?}", nh);
    }
    if let Some(nkh) = gguf.kv.get("llama.attention.head_count_kv") {
        println!("n_kv_heads: {:?}", nkh);
    }
    if let Some(inter) = gguf.kv.get("llama.feed_forward_length") {
        println!("intermediate_size: {:?}", inter);
    }
}
