// Simple end-to-end test comparing embedding layer output

use infer_train::model::GgufFile;
use infer_train::backend::arm::vec_dot::vec_dot_q6_k_q8_k;
use infer_train::backend::arm::quant::quantize_q8_k;
use infer_train::types::{BlockQ6_K, BlockQ8_K};

fn main() {
    let model_path = "/Users/jia/Desktop/infer_train/Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    // Find token_embd.weight
    let embd_tensor = gguf.tensors.iter().find(|t| t.name == "token_embd.weight").expect("No token_embd");
    println!("token_embd.weight: dims={:?}, quant={:?}", embd_tensor.dims, embd_tensor.quant_type);

    // Get tensor data
    let embd_data = gguf.get_tensor_data("token_embd.weight").expect("Failed to get tensor data");
    println!("Embedding data size: {} bytes", embd_data.len());

    // Create a simple hidden state (all ones)
    let hidden_size = 1024;
    let mut hidden = vec![1.0f32; hidden_size];

    // Quantize hidden state to Q8_K
    let nb = (hidden_size + 255) / 256;
    let mut q8k_blocks: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; nb];
    quantize_q8_k(&hidden, &mut q8k_blocks);

    println!("Quantized {} hidden values into {} Q8_K blocks", hidden_size, nb);

    // Check first Q8_K block
    println!("\nFirst Q8_K block:");
    println!("  d: {}", q8k_blocks[0].d);
    println!("  qs[0..8]: {:?}", &q8k_blocks[0].qs[0..8]);
    println!("  bsums[0..4]: {:?}", &q8k_blocks[0].bsums[0..4]);

    // Try to compute dot product with first column of embedding
    // embedding dims: [hidden_size, vocab_size] = [1024, 151936]
    let vocab_size = 151936;

    // Get first column (token 0)
    let col_blocks: Vec<BlockQ6_K> = unsafe {
        let ptr = embd_data.as_ptr() as *const BlockQ6_K;
        std::slice::from_raw_parts(ptr, nb).to_vec()
    };

    println!("\nFirst column (token 0) embedding:");
    println!("  Number of Q6_K blocks: {}", col_blocks.len());
    println!("  First Q6_K block d: {:?}", col_blocks[0].d);
    println!("  First Q6_K block scales[0..4]: {:?}", &col_blocks[0].scales[0..4]);

    // Compute dot product
    let dot = unsafe { vec_dot_q6_k_q8_k(&col_blocks, &q8k_blocks) };
    println!("\nDot product (hidden · embd[:, 0]): {}", dot);

    // Compare with expected (we can't know the exact expected value, but we can check if it's reasonable)
    if dot.is_nan() || dot.is_infinite() {
        println!("ERROR: Invalid result!");
    } else {
        println!("Result is a valid float");
    }

    // Try a few more tokens
    println!("\n=== Testing multiple tokens ===");
    for token_id in [0, 1, 100, 1000, 151935].iter() {
        let offset = token_id * 1024 / 256 * std::mem::size_of::<BlockQ6_K>();
        let col_blocks: Vec<BlockQ6_K> = unsafe {
            let ptr = embd_data.as_ptr().add(*token_id as usize * std::mem::size_of::<BlockQ6_K>()) as *const BlockQ6_K;
            std::slice::from_raw_parts(ptr, nb).to_vec()
        };
        let dot = unsafe { vec_dot_q6_k_q8_k(&col_blocks, &q8k_blocks) };
        println!("Token {}: dot = {}", token_id, dot);
    }
}
