// Check quantization types for attention weights
// Run with: cargo run --bin check_quant <model.gguf>

use infer_train::model::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    println!("=== Attention Q weights by layer ===");
    for tensor in &gguf.tensors {
        if tensor.name.contains("attn_q.weight") {
            println!(
                "{}: {:?} dims={:?}",
                tensor.name, tensor.quant_type, tensor.dims
            );
        }
    }

    println!("\n=== Attention K weights by layer ===");
    for tensor in &gguf.tensors {
        if tensor.name.contains("attn_k.weight") {
            println!(
                "{}: {:?} dims={:?}",
                tensor.name, tensor.quant_type, tensor.dims
            );
        }
    }

    println!("\n=== Attention V weights by layer ===");
    for tensor in &gguf.tensors {
        if tensor.name.contains("attn_v.weight") {
            println!(
                "{}: {:?} dims={:?}",
                tensor.name, tensor.quant_type, tensor.dims
            );
        }
    }

    println!("\n=== Attention norm weights by layer ===");
    for tensor in &gguf.tensors {
        if tensor.name.ends_with("attn_norm.weight") {
            println!(
                "{}: {:?} dims={:?}",
                tensor.name, tensor.quant_type, tensor.dims
            );
        }
    }

    println!("\n=== Q/K norm weights by layer ===");
    for tensor in &gguf.tensors {
        if tensor.name.contains("attn_q_norm.weight")
            || tensor.name.contains("attn_k_norm.weight")
        {
            println!(
                "{}: {:?} dims={:?}",
                tensor.name, tensor.quant_type, tensor.dims
            );
        }
    }

    println!("\n=== FFN norm weights by layer ===");
    for tensor in &gguf.tensors {
        if tensor.name.ends_with("ffn_norm.weight") {
            println!(
                "{}: {:?} dims={:?}",
                tensor.name, tensor.quant_type, tensor.dims
            );
        }
    }
    
    println!("\n=== FFN gate weights (L0, L1, L2) ===");
    for tensor in &gguf.tensors {
        if tensor.name == "blk.0.ffn_gate.weight" || tensor.name == "blk.1.ffn_gate.weight" || tensor.name == "blk.2.ffn_gate.weight" {
            println!("\n{}: {:?}", tensor.name, tensor.quant_type);
            println!("  dims: {:?}", tensor.dims);
            
            if let Some(data) = gguf.get_tensor_data(&tensor.name) {
                println!("  data len: {} bytes", data.len());
                
                match tensor.quant_type {
                    infer_train::types::QuantType::Q5_K => {
                        let blocks: &[infer_train::types::BlockQ5_K] = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const infer_train::types::BlockQ5_K, 3.min(data.len() / 176))
                        };
                        for (i, block) in blocks.iter().enumerate() {
                            let d = half::f16::from_bits(block.d).to_f32();
                            let dmin = half::f16::from_bits(block.dmin).to_f32();
                            println!("  Block {}: d={:.4} dmin={:.4}", i, d, dmin);
                            println!("    scales[0..4]: {:?}", &block.scales[..4]);
                        }
                    }
                    _ => {
                        println!("  Quant type not handled");
                    }
                }
            }
        }
    }

    // Count quant types
    let mut q_types: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for tensor in &gguf.tensors {
        if tensor.name.contains("attn_q.weight") {
            *q_types.entry(format!("{:?}", tensor.quant_type)).or_insert(0) +=
                1;
        }
    }
    println!("\n=== Q weight quant type distribution ===");
    for (t, count) in &q_types {
        println!("  {}: {} layers", t, count);
    }
}