// Check weight layout by reading actual memory
use infer_train::model::GgufFile;

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or("Qwen3-0.6B-UD-Q4_K_XL.gguf".to_string());
    let gguf = GgufFile::load(&model_path).expect("Failed to load model");

    // Find attn_q weight
    let tensor = gguf.tensors.iter()
        .find(|t| t.name.contains("attn_q.weight"))
        .expect("attn_q.weight not found");

    println!("Tensor: {}", tensor.name);
    println!("Dims: {:?}", tensor.dims);
    println!("Strides: {:?}", tensor.strides);
    println!("Quant type: {:?}", tensor.quant_type);

    // Calculate expected strides
    let ne0 = tensor.dims.get(0).copied().unwrap_or(0);
    let ne1 = tensor.dims.get(1).copied().unwrap_or(0);
    let nb0 = tensor.strides.get(0).copied().unwrap_or(0);
    let nb1 = tensor.strides.get(1).copied().unwrap_or(0);

    println!("\nDimensions:");
    println!("  ne0 (dim 0) = {}", ne0);
    println!("  ne1 (dim 1) = {}", ne1);

    // Block info
    let block_size = match tensor.quant_type {
        infer_train::types::QuantType::Q5_K => 176,
        infer_train::types::QuantType::Q4_K => 144,
        infer_train::types::QuantType::Q6_K => 210,
        _ => 64,
    };
    let elements_per_block = 256;

    println!("\nBlock info:");
    println!("  Block size: {} bytes", block_size);
    println!("  Elements per block: {}", elements_per_block);

    // Calculate expected layout
    let blocks_per_input = ne0 / elements_per_block;
    let blocks_per_output = ne1;  // Each output row has blocks_per_input blocks

    println!("\nExpected layouts:");
    println!("  If [output][blocks_per_input]:");
    println!("    nb1 = blocks_per_input * block_size = {}", blocks_per_input * block_size);
    println!("  If [blocks_per_input][output] (interleaved):");
    println!("    nb1 = block_size = {}", block_size);
    println!("  If output features are interleaved (4 at a time):");
    println!("    nb1 = 4 * block_size = {}", 4 * block_size);

    println!("\nActual nb1 = {}", nb1);

    // Determine layout
    if nb1 == (blocks_per_input * block_size) as u64 {
        println!("\nLayout: Contiguous per output row");
    } else if nb1 == block_size as u64 {
        println!("\nLayout: Interleaved (one block per input dimension per output)");
    } else if nb1 == (4 * block_size) as u64 {
        println!("\nLayout: 4-output-feature interleaved");
        println!("  Each group of 4 output features shares input blocks");
    } else {
        println!("\nLayout: Unknown (nb1 doesn't match any expected pattern)");
    }

    // Now check the actual data to understand the layout
    let data = gguf.get_tensor_data(&tensor.name).expect("Failed to get tensor data");

    // For Q5_K, each block is 176 bytes representing 256 elements
    // Let's check the first few blocks to see the pattern
    println!("\nFirst few bytes of data (hex):");
    for i in 0..std::cmp::min(64, data.len()) {
        print!("{:02x} ", data[i]);
        if (i + 1) % 16 == 0 {
            println!();
        }
    }

    // Try to understand the pattern
    // If interleaved by 4 output features:
    // First 4 blocks would be: output 0-3, all at input block 0
    // Next 4 blocks would be: output 4-7, all at input block 0
    // ...
    // After 512 blocks (2048/4 = 512 groups), we get to input block 1

    // If contiguous per output:
    // First 4 blocks would be: output 0, input blocks 0-3
    // Next 4 blocks would be: output 1, input blocks 0-3
    // ...

    println!("\n\nTotal blocks: {}", data.len() / block_size);
    let total_blocks = data.len() / block_size;
    let expected_blocks_contiguous = ne1 * (ne0 / elements_per_block);
    println!("Expected blocks (contiguous per output): {}", expected_blocks_contiguous);

    if total_blocks == expected_blocks_contiguous as usize {
        println!("Block count matches contiguous layout");
    } else {
        println!("Block count mismatch!");
    }
}
