use infer_train::model::GgufFile;
use infer_train::types::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let gguf = GgufFile::load(&args[1]).expect("Failed to load GGUF");

    println!("=== Tensor Layout Analysis ===\n");

    let patterns = ["attn_q.weight", "attn_k.weight", "ffn_gate.weight"];

    for pattern in &patterns {
        if let Some(t) = gguf.tensors.iter().find(|t| t.name.contains(pattern)) {
            println!("--- {} ---", t.name);

            let ne0 = t.dims[0] as u64;
            let ne1 = t.dims[1] as u64;
            let nb0 = t.strides[0] as u64;
            let nb1 = t.strides[1] as u64;

            let (block_size, block_elements) = match t.quant_type {
                QuantType::Q5_K => (176, 256),
                QuantType::Q4_K => (144, 256),
                QuantType::Q6_K => (210, 256),
                QuantType::Q4_0 => (18, 32),
                _ => (nb0 as usize, 1),
            };

            let n_blocks_ne0 = (ne0 + block_elements - 1) / block_elements;
            let expected_contiguous = n_blocks_ne0 * block_size as u64;
            let interleaving = if block_size > 0 { nb1 / block_size as u64 } else { 0 };

            println!("  ne=[{}, {}], nb=[{}, {}]", ne0, ne1, nb0, nb1);
            println!("  block_size={}, n_blocks={}", block_size, n_blocks_ne0);
            println!("  expected nb[1] for contiguous: {}", expected_contiguous);
            println!("  actual nb[1]: {}", nb1);

            if nb1 == expected_contiguous {
                println!("  => CONTIGUOUS layout");
            } else if interleaving > 0 && nb1 % block_size as u64 == 0 {
                println!("  => INTERLEAVED with factor {}", interleaving);
            }
            println!();
        }
    }
}
