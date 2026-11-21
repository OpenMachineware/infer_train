use infer_train::model::GgufFile;
use infer_train::types::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = if args.len() > 1 {
        args[1].as_str()
    } else {
        "Qwen3-0.6B-UD-Q4_K_XL.gguf"
    };

    let gguf = GgufFile::load(path).unwrap();

    println!("=== Key Weight Tensors ===");
    for tensor in &gguf.tensors {
        if tensor.name.starts_with("blk.0.")
            && (tensor.name.contains("attn_q")
                || tensor.name.contains("attn_k")
                || tensor.name.contains("attn_v")
                || tensor.name.contains("ffn_gate"))
        {
            let block_size = match tensor.quant_type {
                QuantType::Q4_K => std::mem::size_of::<BlockQ4_K>(),
                QuantType::Q5_K => std::mem::size_of::<BlockQ5_K>(),
                QuantType::Q6_K => std::mem::size_of::<BlockQ6_K>(),
                QuantType::IQ4_XS => std::mem::size_of::<BlockIQ4XS>(),
                _ => 0,
            };

            println!("\n{}:", tensor.name);
            println!("  dims = {:?}", tensor.dims);
            println!("  strides = {:?} bytes", tensor.strides);
            println!("  quant_type = {:?}", tensor.quant_type);
            println!("  block_size = {} bytes", block_size);

            if tensor.dims.len() == 2 {
                let in_features = tensor.dims[0] as usize;
                let out_features = tensor.dims[1] as usize;
                println!(
                    "  in_features = {}, out_features = {}",
                    in_features, out_features
                );

                let n_blocks = (in_features + 255) / 256;
                println!(
                    "  n_blocks per column = {} (in_features / 256)",
                    n_blocks
                );

                let expected_col_size = n_blocks * block_size;
                println!(
                    "  expected column size = {} bytes (n_blocks * block_size)",
                    expected_col_size
                );

                let stride1 = tensor.strides[1] as usize;
                println!("  actual stride[1] = {} bytes", stride1);

                if stride1 == expected_col_size {
                    println!(
                        "  ✓ stride[1] matches expected column size (weights are COLUMN-MAJOR)"
                    );
                } else {
                    println!("  ✗ MISMATCH: stride[1] != expected column size");
                    println!(
                        "  stride[1] / block_size = {} blocks",
                        stride1 / block_size
                    );
                }
            }
        }
    }

    println!("\n=== Block sizes ===");
    println!("BlockQ4_K = {} bytes", std::mem::size_of::<BlockQ4_K>());
    println!("BlockQ5_K = {} bytes", std::mem::size_of::<BlockQ5_K>());
    println!("BlockQ6_K = {} bytes", std::mem::size_of::<BlockQ6_K>());
    println!("BlockIQ4XS = {} bytes", std::mem::size_of::<BlockIQ4XS>());
}
