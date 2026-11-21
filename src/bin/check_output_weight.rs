use infer_train::model::GgufFile;

fn main() {
    let path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(path).unwrap();

    println!("=== Output Weight Analysis ===\n");

    for tensor in &gguf.tensors {
        if tensor.name == "output.weight" || tensor.name == "lm_head.weight" {
            println!("Tensor: {}", tensor.name);
            println!("  dims = {:?}", tensor.dims);
            println!("  strides = {:?}", tensor.strides);
            println!("  quant_type = {:?}", tensor.quant_type);

            if tensor.dims.len() == 2 {
                let in_features = tensor.dims[0] as usize;
                let out_features = tensor.dims[1] as usize;
                println!("  in_features (hidden_size) = {}", in_features);
                println!("  out_features (vocab_size) = {}", out_features);

                println!(
                    "\n  For y = W * x with W[{}, {}] in column-major:",
                    in_features, out_features
                );
                println!("    Computing y[0..151936]");
                println!(
                    "    Each y[j] reads column j at offset j * stride[1]"
                );

                let stride1 = tensor.strides[1] as usize;
                println!("\n  stride[1] = {} bytes", stride1);
                println!(
                    "  Total weight bytes = {} MB",
                    stride1 * out_features / 1024 / 1024
                );

                println!("\n  Memory access pattern:");
                println!("    y[0] reads bytes [0, {})", stride1);
                println!("    y[1] reads bytes [{}, {})", stride1, 2 * stride1);
                println!("    ...");
                println!(
                    "    y[151935] reads bytes [{}, {})",
                    151935 * stride1,
                    151936 * stride1
                );

                println!(
                    "\n  This IS sequential! Each column is read sequentially."
                );
            }
        }
    }

    // Check token_embd for comparison
    println!("\n=== Token Embedding Analysis ===\n");
    for tensor in &gguf.tensors {
        if tensor.name == "token_embd.weight" {
            println!("Tensor: {}", tensor.name);
            println!("  dims = {:?}", tensor.dims);
            println!("  strides = {:?}", tensor.strides);
            println!("  quant_type = {:?}", tensor.quant_type);
        }
    }
}
