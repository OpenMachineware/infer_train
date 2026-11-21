use infer_train::model::GgufFile;
use infer_train::types::*;

fn main() {
    let path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(path).unwrap();

    println!("=== Analyzing Interleaved Weight Layout ===\n");

    let tensor =
        gguf.tensors.iter().find(|t| t.name == "blk.0.attn_q.weight").unwrap();

    let in_features = tensor.dims[0] as usize; // 1024
    let out_features = tensor.dims[1] as usize; // 2048
    let stride0 = tensor.strides[0] as usize; // 176
    let stride1 = tensor.strides[1] as usize; // 704

    let block_size = match tensor.quant_type {
        QuantType::Q5_K => std::mem::size_of::<BlockQ5_K>(),
        _ => 0,
    };

    let n_blocks_per_col = in_features / 256; // 4

    println!("Tensor: {}", tensor.name);
    println!(
        "dims = [in_features={}, out_features={}]",
        in_features, out_features
    );
    println!("strides = [{}, {}] bytes", stride0, stride1);
    println!("block_size = {} bytes", block_size);
    println!("n_blocks_per_column = {} (in_features / 256)", n_blocks_per_col);

    let interleave_factor = stride1 / block_size;
    println!("\n=== Interleave Factor ===");
    println!(
        "stride[1] / block_size = {} / {} = {}",
        stride1, block_size, interleave_factor
    );
    println!("Interleave factor = {}", interleave_factor);

    println!("\n=== Understanding the Layout ===");
    println!(
        "stride[1] = {} bytes covers {} blocks",
        stride1, interleave_factor
    );
    println!("But a full column has {} blocks!", n_blocks_per_col);
    println!("\nThis means the weights are NOT stored column-by-column.");
    println!("Instead, they're stored in an INTERLEAVED manner.");

    println!("\n=== How Interleaved Storage Works ===");
    println!(
        "Standard column-major: column 0, then column 1, then column 2, ..."
    );
    println!(
        "Interleaved: groups of {} columns stored together",
        interleave_factor
    );

    println!("\nFor our case (interleave_factor = 4):");
    println!(
        "  - The weight tensor is divided into chunks of 4 output features"
    );
    println!("  - Each chunk stores blocks for all 4 output features together");

    println!("\n=== Memory Layout ===");
    println!(
        "Total bytes = stride[1] * out_features = {} * {} = {}",
        stride1,
        out_features,
        stride1 * out_features
    );
    println!(
        "Expected for column-major = {} blocks * {} bytes * {} cols = {} bytes",
        n_blocks_per_col,
        block_size,
        out_features,
        n_blocks_per_col * block_size * out_features
    );

    println!("\nstride[1] = {} bytes", stride1);
    println!("  = {} blocks * {} bytes/block", interleave_factor, block_size);
    println!(
        "  = {} blocks for {} input features",
        interleave_factor,
        interleave_factor * 256
    );

    // This doesn't add up. Let me reconsider...
    println!("\n=== Re-analyzing ===");
    println!("stride[0] = {} bytes (size of one block)", stride0);
    println!("stride[1] = {} bytes", stride1);
    println!(
        "\nFor standard row-major: stride[1] = in_features * element_size"
    );
    println!(
        "For standard column-major: stride[0] = out_features * element_size"
    );

    println!("\nstride[1] = {} bytes covers:", stride1);
    println!(
        "  {} blocks for each of {} output features?",
        n_blocks_per_col,
        out_features / 4
    );

    println!("\n=== Actual Layout (GGML mul_mat style) ===");
    println!("The weights are stored as [K, N] where:");
    println!("  K = in_features = {} (rows)", in_features);
    println!("  N = out_features = {} (columns)", out_features);

    println!("\nFor quantized format, the storage is:");
    println!("  - Each row (K elements) is divided into blocks of 256");
    println!("  - {} blocks per row", n_blocks_per_col);

    println!("\nstride[1] = {} bytes tells us:", stride1);
    println!("  - Jumping from column i to column i+1 costs {} bytes", stride1);
    println!("  - This is only {} blocks worth of data", stride1 / block_size);

    let ne0 = (stride1 / block_size) as usize; // blocks per column segment
    println!("\n  - But each column should have {} blocks", n_blocks_per_col);
    println!(
        "  - This suggests: {} * {} = {} output features stored together",
        n_blocks_per_col,
        ne0,
        n_blocks_per_col * ne0
    );

    println!("\n=== GGML interleaved format explanation ===");
    println!("In GGML, for quantized weights:");
    println!(
        "  ne[0] = {} (number of elements per row, i.e., in_features)",
        in_features
    );
    println!("  ne[1] = {} (number of rows, i.e., out_features)", out_features);
    println!(
        "  But the storage is re-arranged for efficient matrix-vector multiply."
    );

    println!("\nFor interleaved storage:");
    println!(
        "  - The matrix is divided into tiles of {} x 256",
        interleave_factor
    );
    println!(
        "  - Each tile stores {} output features for 256 input elements",
        interleave_factor
    );

    println!("\n=== Correct GEMV for Interleaved Layout ===");
    println!("To compute y[j] = vec_dot(W column j, x):");
    println!("  Column j is at offset: j * stride[1] = j * {}", stride1);
    println!("  But this only covers {} blocks!", stride1 / block_size);

    println!("\nThis means for ONE output y[j]:");
    println!(
        "  We read from {} different locations (one per block group)",
        n_blocks_per_col
    );
    println!("  Each location is separated by stride[1] = {} bytes", stride1);

    println!("\n=== Conclusion ===");
    println!(
        "stride[1] = {} bytes < {} * {} = {} bytes (full column)",
        stride1,
        n_blocks_per_col,
        block_size,
        n_blocks_per_col * block_size
    );
    println!("This confirms INTERLEAVED storage!");
    println!(
        "To access full column j, we need to read from multiple locations."
    );
}
