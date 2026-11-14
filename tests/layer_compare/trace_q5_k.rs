// Trace Q5_K x Q8_K vec_dot computation for debugging
use infer_train::types::*;
use infer_train::backend::arm::{vec_dot, quant};
use infer_train::model::GgufFile;

fn main() {
    let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";
    let gguf = GgufFile::load(model_path).expect("Failed to load GGUF");

    // Get L2 FFN gate weights
    let gate_name = "blk.2.ffn_gate.weight";
    let gate_data = gguf.get_tensor_data(gate_name).unwrap();
    let gate_tensor = gguf.tensors.iter().find(|t| t.name == gate_name).unwrap();
    println!("=== {} ===", gate_name);
    println!("Quant type: {:?}", gate_tensor.quant_type);
    println!("Dims: {:?}", gate_tensor.dims);

    let gate_blocks: &[BlockQ5_K] = unsafe {
        std::slice::from_raw_parts(gate_data.as_ptr() as *const BlockQ5_K, gate_data.len() / 176)
    };

    // Check first few blocks
    for i in 0..5 {
        let d = fp16_to_fp32(gate_blocks[i].d);
        let dmin = fp16_to_fp32(gate_blocks[i].dmin);

        // Extract sub-block scales (same logic as vec_dot)
        const KMASK1: u32 = 0x3f3f3f3f;
        const KMASK2: u32 = 0x0f0f0f0f;
        const KMASK3: u32 = 0x03030303;

        let mut utmp: [u32; 4] = [0; 4];
        utmp[0..3].copy_from_slice(unsafe {
            std::slice::from_raw_parts(gate_blocks[i].scales.as_ptr() as *const u32, 3)
        });
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let scales: [u8; 8] = unsafe {
            std::slice::from_raw_parts(utmp[0..2].as_ptr() as *const u8, 8)
                .try_into()
                .unwrap()
        };

        println!("Block {}: d={:.6} dmin={:.6} scales={:?}", i, d, dmin, scales);
    }

    // Now test with actual input
    // Create a simple Q8_K input (matching what we'd get from RMS norm)
    let test_input = vec![1.0f32; 1024];  // Simple normalized input
    let nb = (1024 + QK_K - 1) / QK_K;
    let mut input_q8k: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; nb];
    quant::quantize_q8_k(&test_input, &mut input_q8k);

    println!("\n=== Test with input = [1.0; 1024] ===");
    println!("Q8_K block 0: d={:.4}", input_q8k[0].d);
    println!("Q8_K block 0 first 10 qs: {:?}", &input_q8k[0].qs[..10]);

    // Compute output for first 10 rows
    println!("\n=== Outputs for first 10 rows ===");
    for row in 0..10 {
        let row_blocks = &gate_blocks[row * 4..(row + 1) * 4];
        let result = vec_dot::vec_dot_q5_k_q8_k(row_blocks, &input_q8k);
        println!("Row {}: {:.4}", row, result);
    }

    // Also test with input = [0.1; 1024] (closer to actual normalized values)
    let test_input2 = vec![0.1f32; 1024];
    let mut input_q8k2: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; nb];
    quant::quantize_q8_k(&test_input2, &mut input_q8k2);

    println!("\n=== Test with input = [0.1; 1024] ===");
    println!("Q8_K block 0: d={:.4}", input_q8k2[0].d);

    println!("\n=== Outputs for first 10 rows ===");
    for row in 0..10 {
        let row_blocks = &gate_blocks[row * 4..(row + 1) * 4];
        let result = vec_dot::vec_dot_q5_k_q8_k(row_blocks, &input_q8k2);
        println!("Row {}: {:.4}", row, result);
    }

    // Test with actual L2-FFN-normed values (mix of large positive and negative)
    let mut test_input3 = vec![0.0f32; 1024];
    // Create a mix similar to what we see in L2-FFN-normed
    for i in 0..512 {
        test_input3[i] = 2.0;  // Some positive
    }
    for i in 512..1024 {
        test_input3[i] = -10.0;  // Some negative (matching min=-17.1)
    }

    let mut input_q8k3: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; nb];
    quant::quantize_q8_k(&test_input3, &mut input_q8k3);

    println!("\n=== Test with mixed input (matching L2-FFN-normed) ===");
    println!("Input: first 512 = 2.0, last 512 = -10.0");
    println!("Q8_K block 0: d={:.4}", input_q8k3[0].d);
    println!("Q8_K block 0 first 10 qs: {:?}", &input_q8k3[0].qs[..10]);

    println!("\n=== Outputs for first 10 rows ===");
    for row in 0..10 {
        let row_blocks = &gate_blocks[row * 4..(row + 1) * 4];
        let result = vec_dot::vec_dot_q5_k_q8_k(row_blocks, &input_q8k3);
        println!("Row {}: {:.4}", row, result);
    }

    // Test with actual L2-FFN-normed sample values from forward pass
    // L2-FFN-normed_sample10: [-0.049, 0.00009, 1.365, -0.15, 0.853, 0.139, 0.224, -1.72, 0.278, -0.78]
    let mut test_input4 = vec![-17.1135f32; 1024];  // Fill with min value to maximize scale
    test_input4[0] = -0.049;
    test_input4[1] = 0.00009;
    test_input4[2] = 1.365;
    test_input4[3] = -0.15;
    test_input4[4] = 0.853;
    test_input4[5] = 0.139;
    test_input4[6] = 0.224;
    test_input4[7] = -1.72;
    test_input4[8] = 0.278;
    test_input4[9] = -0.78;
    // Keep rest as -17.1135 (the min value)

    let mut input_q8k4: Vec<BlockQ8_K> = vec![unsafe { std::mem::zeroed() }; nb];
    quant::quantize_q8_k(&test_input4, &mut input_q8k4);

    println!("\n=== Test with actual L2-FFN-normed values (min=-17.1) ===");
    println!("Q8_K block 0: d={:.4}", input_q8k4[0].d);
    println!("Q8_K block 0 first 10 qs: {:?}", &input_q8k4[0].qs[..10]);

    println!("\n=== Outputs for first 10 rows ===");
    for row in 0..10 {
        let row_blocks = &gate_blocks[row * 4..(row + 1) * 4];
        let result = vec_dot::vec_dot_q5_k_q8_k(row_blocks, &input_q8k4);
        println!("Row {}: {:.4}", row, result);
    }

    // Compare with Q8_K scale from actual forward pass
    println!("\n=== Scale comparison ===");
    println!("Actual L2-FFN-input_Q8_K scale: 0.1348");
    println!("Test scale with min=-17.1: {:.4}", input_q8k4[0].d);
}
