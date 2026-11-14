// Dump L7 and L8 attention Q weights for debugging
use infer_train::model::GgufFile;
use infer_train::types::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    for layer_idx in [0, 1, 2, 7, 8] {
        for tensor_suffix in ["attn_q.weight", "attn_v.weight", "attn_output.weight", "ffn_down.weight"].iter() {
            println!("\n=== Layer {} {} ===", layer_idx, tensor_suffix);
            let weight_name = format!("blk.{}.{}", layer_idx, tensor_suffix);

            if let Some(tensor) = gguf.tensors.iter().find(|t| t.name == weight_name) {
                println!("Tensor: {}", tensor.name);
                println!("Quant type: {:?}", tensor.quant_type);
                println!("Dims: {:?}", tensor.dims);

                if let Some(data) = gguf.get_tensor_data(&tensor.name) {
                    println!("Data size: {} bytes", data.len());
                    println!("First 20 bytes: {:02x?}", &data[..20.min(data.len())]);

                    match tensor.quant_type {
                        QuantType::Q4_K => {
                            // Dump first few Q4_K blocks
                            let blocks: &[BlockQ4_K] = unsafe {
                                std::slice::from_raw_parts(data.as_ptr() as *const BlockQ4_K, 5.min(data.len() / 144))
                            };
                            for (i, block) in blocks.iter().enumerate() {
                                let d = half::f16::from_bits(block.d).to_f32();
                                let dmin = half::f16::from_bits(block.dmin).to_f32();
                                println!("  Block {}: d={:.6} (bits 0x{:04x}) dmin={:.6}", i, d, block.d, dmin);
                                println!("    scales: {:?}", &block.scales[..8]);
                            }
                        }
                        QuantType::IQ4_XS => {
                            // Dump first few IQ4_XS blocks
                            let blocks: &[BlockIQ4XS] = unsafe {
                                std::slice::from_raw_parts(data.as_ptr() as *const BlockIQ4XS, 5.min(data.len() / 136))
                            };
                            for (i, block) in blocks.iter().enumerate() {
                                let d = half::f16::from_bits(block.d).to_f32();
                                println!("  Block {}: d={:.6} (bits 0x{:04x}) scales_h={}", i, d, block.d, block.scales_h);
                                println!("    scales_l: {:?}", &block.scales_l[..4]);
                            }
                        }
                        QuantType::Q6_K => {
                            // Dump first few Q6_K blocks
                            let blocks: &[BlockQ6_K] = unsafe {
                                std::slice::from_raw_parts(data.as_ptr() as *const BlockQ6_K, 5.min(data.len() / 210))
                            };
                            for (i, block) in blocks.iter().enumerate() {
                                let d = half::f16::from_bits(block.d).to_f32();
                                println!("  Block {}: d={:.6} (bits 0x{:04x})", i, d, block.d);
                            }
                        }
                        _ => {
                            println!("  Quant type not handled for dumping");
                        }
                    }
                }
            }
        }
    }
}
