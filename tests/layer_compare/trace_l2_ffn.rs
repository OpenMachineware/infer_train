// Trace L2 FFN gate computation with actual model forward values
use infer_train::types::*;
use infer_train::backend::arm::{vec_dot, quant};
use infer_train::model::GgufFile;
use infer_train::inference::LlamaModel;

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");
    let mut model = LlamaModel::new(gguf).expect("Failed to create model");

    // Run forward pass for token 27 (first token) to capture L2 state
    let _ = model.forward_token(27);

    // Now manually inspect L2 FFN computation
    println!("=== L2 FFN Debug ===");

    // Get L2 FFN gate weights
    let gate_name = "blk.2.ffn_gate.weight";
    let gate_data = model.model.gguf.get_tensor_data(gate_name).unwrap();
    let gate_blocks: &[BlockQ5_K] = unsafe {
        std::slice::from_raw_parts(gate_data.as_ptr() as *const BlockQ5_K, gate_data.len() / 176)
    };

    // Check actual L2 FFN-normed values
    // We need to compute them ourselves
    // Let's manually compute L0->L1->L2 forward to capture the state

    // For now, let's just test with known input range
    let test_input = vec![-17.0f32, 2.5, 0.0, 1.0, -1.0];
    println!("Test input: {:?}", test_input);

    // Quantize to Q8_K
    let mut full_input = vec![0.0f32; 1024];
    for (i, &v) in test_input.iter().enumerate() {
        if i < 1024 {
            full_input[i] = v;
        }
    }

    let nb = (1024 + QK_K - 1) / QK_K;
    let mut input_q8k: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; nb];
    quant::quantize_q8_k(&full_input, &mut input_q8k);

    println!("Q8_K scale: {:.4}", input_q8k[0].d);

    // Compute output for first few rows
    println!("\n=== Gate outputs ===");
    for row in 0..5 {
        let row_blocks = &gate_blocks[row * 4..(row + 1) * 4];
        let result = vec_dot::vec_dot_q5_k_q8_k(row_blocks, &input_q8k);
        println!("Row {}: {:.4}", row, result);
    }
}
