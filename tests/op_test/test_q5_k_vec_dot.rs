// Test Q5_K GEMV with real model weights
use infer_train::model::GgufFile;
use infer_train::types::*;
use infer_train::backend::arm::{vec_dot, quant};

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");
    
    // Get the L0 and L2 FFN gate weights
    for layer_idx in [0, 2] {
        let tensor_name = format!("blk.{}.ffn_gate.weight", layer_idx);
        println!("\n=== {} ===", tensor_name);
        
        if let Some(data) = gguf.get_tensor_data(&tensor_name) {
            let tensor = gguf.tensors.iter().find(|t| t.name == tensor_name).unwrap();
            println!("Quant type: {:?}", tensor.quant_type);
            println!("Dims: {:?}", tensor.dims);
            
            // Create a simple input vector
            let input = vec![0.1f32; 1024];  // All 0.1
            
            // Quantize to Q8_K
            let nb = (1024 + QK_K - 1) / QK_K;
            let mut input_q8k: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; nb];
            quant::quantize_q8_k(&input, &mut input_q8k);
            
            println!("Input: all 0.1");
            println!("Q8_K d = {}, first qs = {:?}", input_q8k[0].d, &input_q8k[0].qs[..10]);
            
            // Compute first few output elements using Q5_K
            let weights: &[BlockQ5_K] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const BlockQ5_K, 3072 * 4)  // 3072 rows * 4 blocks each
            };
            
            let blocks_per_row = 4;  // 1024 / 256 = 4
            
            let mut outputs = vec![0.0f32; 10];
            for i in 0..10 {
                let row_blocks = &weights[i * blocks_per_row..(i + 1) * blocks_per_row];
                outputs[i] = vec_dot::vec_dot_q5_k_q8_k(row_blocks, &input_q8k);
            }
            
            println!("First 10 outputs: {:?}", outputs);
            
            // Compute expected value
            // With input = 0.1, Q8_K d ≈ 0.1/127 ≈ 0.00079
            // Each row computes sum of 1024 elements
            // With scale around 0.0001 and weight values around 8-15...
            // This is complex. Let me just print the actual results.
        }
    }
}