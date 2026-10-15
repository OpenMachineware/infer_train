// Check output tensor layout

use infer_train::model::{
    GgufFile, ModelConfig, tensor::Tensor, weights::WeightStore,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("Loading model: {}", model_path);

    let store = WeightStore::from_gguf(model_path).unwrap();
    let gguf = infer_train::model::GgufFile::open(model_path).unwrap();

    // Check output tensor
    if let Some(output) = store.get("output.weight") {
        println!("\n=== Output tensor ===");
        println!("Shape: {:?}", output.shape);
        println!("Dtype: {:?}", output.dtype);
        println!("Bytes: {}", output.bytes().len());

        // Calculate expected bytes
        let total_elems: usize = output.shape.iter().product();
        println!("Total elements: {}", total_elems);

        // For Q6_K: each block has 256 elements
        let block_size = 256;
        let num_blocks = (total_elems + block_size - 1) / block_size;
        println!("Num blocks (256 per block): {}", num_blocks);

        // Check block alignment
        if total_elems % block_size != 0 {
            println!(
                "WARNING: Total elements {} is not divisible by block_size {}",
                total_elems, block_size
            );
            println!("  Remainder: {}", total_elems % block_size);
        }

        // Check Q6_K block size
        let q6k_block_bytes =
            std::mem::size_of::<infer_train::quant::types::BlockQ6K>();
        println!("Q6_K block size: {} bytes", q6k_block_bytes);
        let expected_bytes = num_blocks * q6k_block_bytes;
        println!(
            "Expected bytes (num_blocks * block_size): {}",
            expected_bytes
        );
        println!("Actual bytes: {}", output.bytes().len());

        if output.bytes().len() == expected_bytes {
            println!("✓ Bytes match!");
        } else {
            println!(
                "✗ Bytes mismatch: actual {} vs expected {}",
                output.bytes().len(),
                expected_bytes
            );
        }

        // Check shape interpretation
        println!("\n=== Shape interpretation ===");
        if output.shape.len() == 2 {
            let rows = output.shape[0];
            let cols = output.shape[1];
            println!("Rows (hidden_size): {}", rows);
            println!("Cols (vocab_size): {}", cols);

            // For transposed output: [hidden_size, vocab_size]
            let blocks_per_row = cols / block_size;
            let row_remainder = cols % block_size;
            println!("Blocks per row: {}", blocks_per_row);
            println!("Row remainder: {}", row_remainder);

            // Expected layout
            let expected_total_blocks = rows * blocks_per_row;
            if row_remainder != 0 {
                println!(
                    "WARNING: vocab_size {} not aligned to block_size {}",
                    cols, block_size
                );
            }
        }
    } else {
        println!("Output tensor not found");
    }
}
