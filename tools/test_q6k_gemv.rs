// SPDX-License-Identifier: AGPL-3.0-or-later
// Test Q6_K GEMV with actual model weights

use infer_train::inference::{Device, InferenceContext};
use infer_train::quant::kv_cache::quantize_row_q8_k_neon;
use infer_train::quant::types::QK_K;
use infer_train::quant::vec_dot::arm::vec_dot_q6_k_q8_k_neon;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
        .expect("Failed to load model");

    // Get layer 0 w2 (down projection) which is Q6_K
    let layer = &ctx.model.layers[0];
    let w2 = layer.w2.as_ref().unwrap();

    println!("w2 dtype: {:?}", w2.dtype);
    println!("w2 shape: {:?}", w2.shape);

    // Create test input (simulating SwiGLU output)
    let intermediate_size = w2.shape[0]; // 8960
    let hidden_size = w2.shape[1]; // 1536

    println!("Intermediate size: {}", intermediate_size);
    println!("Hidden size: {}", hidden_size);

    let test_input = vec![0.5f32; intermediate_size];

    // Quantize input to Q8_K
    let nb = intermediate_size / QK_K;
    let mut x_q8k = vec![unsafe { std::mem::zeroed() }; nb];
    quantize_row_q8_k_neon(&test_input, &mut x_q8k);

    println!("Input quantized to {} Q8_K blocks", nb);

    // Test single row GEMV
    use infer_train::quant::types::BlockQ6K;
    let weights: &[BlockQ6K] = unsafe {
        std::slice::from_raw_parts(
            w2.bytes().as_ptr() as *const BlockQ6K,
            hidden_size * nb,
        )
    };

    println!("Weight has {} blocks", weights.len());

    // Compute first row
    let result =
        vec_dot_q6_k_q8_k_neon(intermediate_size, &weights[0..nb], &x_q8k);

    println!("First row result: {:.4}", result);

    // Full GEMV
    let mut output = vec![0.0f32; hidden_size];
    for i in 0..hidden_size {
        output[i] = vec_dot_q6_k_q8_k_neon(
            intermediate_size,
            &weights[i * nb..(i + 1) * nb],
            &x_q8k,
        );
    }

    println!(
        "Output (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        output[0], output[1], output[2], output[3], output[4]
    );

    let all_zeros = output.iter().all(|&x| x == 0.0);
    println!("Output all zeros: {}", all_zeros);

    // Compare with gemv_impl
    let output2 = ctx.model.gemv_impl(&test_input, w2, hidden_size).unwrap();
    println!(
        "gemv_impl output (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        output2[0], output2[1], output2[2], output2[3], output2[4]
    );
}
