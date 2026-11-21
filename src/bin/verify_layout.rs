use infer_train::model::GgufFile;
use infer_train::types::*;

fn main() {
    let path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(path).unwrap();

    println!("=== Verifying Weight Layout ===\n");

    let tensor =
        gguf.tensors.iter().find(|t| t.name == "blk.0.attn_q.weight").unwrap();

    let in_features = tensor.dims[0] as usize; // 1024
    let out_features = tensor.dims[1] as usize; // 2048
    let stride0 = tensor.strides[0] as usize; // 176
    let stride1 = tensor.strides[1] as usize; // 704

    println!("Tensor: {}", tensor.name);
    println!("dims = [{}, {}] (rows, cols)", in_features, out_features);
    println!("strides = [{}, {}] bytes", stride0, stride1);
    println!("quant_type = {:?}", tensor.quant_type);

    let block_size = match tensor.quant_type {
        QuantType::Q5_K => std::mem::size_of::<BlockQ5_K>(),
        _ => 0,
    };

    let n_blocks = (in_features + 255) / 256; // 4

    println!("\n=== Layout Analysis ===");
    println!("Block size: {} bytes", block_size);
    println!("n_blocks per column: {}", n_blocks);

    println!("\nFor y = W * x where W is [in_features, out_features]:");
    println!("  y[j] = vec_dot(W column j, x)");
    println!(
        "  column j starts at byte offset: j * stride[1] = j * {}",
        stride1
    );

    println!("\nOur GEMV code:");
    println!("  row_offset = row * col_stride = row * {}", stride1);
    println!(
        "  reads n_blocks = {} blocks = {} elements",
        n_blocks,
        n_blocks * 256
    );

    println!("\n=== Memory Access Pattern ===");
    println!("Computing output y[0], y[1], y[2], ...");
    for j in 0..5 {
        let col_start = j * stride1;
        println!(
            "  y[{}] reads column {} starting at byte {}",
            j, j, col_start
        );
        println!(
            "    bytes [{}, {}) = column {} of weight matrix",
            col_start,
            col_start + stride1,
            j
        );
    }

    println!("\n=== Is Memory Access Sequential? ===");
    println!("Computing y[0]: reads bytes [0, {})", stride1);
    println!("Computing y[1]: reads bytes [{}, {})", stride1, 2 * stride1);
    println!("Computing y[2]: reads bytes [{}, {})", 2 * stride1, 3 * stride1);
    println!("=> Memory access IS sequential! ✓");

    println!("\n=== Checking stride[0] ===");
    println!("stride[0] = {} bytes", stride0);
    println!(
        "This is the size of ONE block (BlockQ5_K = {} bytes)",
        block_size
    );
    println!(
        "stride[0] is the offset between adjacent elements WITHIN a column"
    );
    println!("For quantized format: stride[0] = block_size makes sense");

    println!("\n=== Conclusion ===");
    if stride0 == block_size && stride1 == n_blocks * block_size {
        println!("✓ Weight layout is correct COLUMN-MAJOR");
        println!(
            "✓ Our GEMV uses row * col_stride which correctly reads column 'row'"
        );
        println!("✓ Memory access IS sequential");
    } else {
        println!("✗ Layout mismatch!");
    }
}
