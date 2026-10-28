// SPDX-License-Identifier: AGPL-3.0-or-later
// Test Q6_K GEMV with actual model weights

use infer_train::inference::Device;
use infer_train::inference::InferenceContext;
use infer_train::quant::kv_cache::quantize_row_q8_k_neon;
use infer_train::quant::types::{BlockQ6K, QK_K};
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

    // Create non-zero test input
    let test_input: Vec<f32> =
        (0..intermediate_size).map(|i| 0.5 + (i % 100) as f32 * 0.01).collect();
    println!(
        "Input (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        test_input[0],
        test_input[1],
        test_input[2],
        test_input[3],
        test_input[4]
    );

    // Quantize input to Q8_K
    let nb = intermediate_size / QK_K;
    let mut x_q8k = vec![unsafe { std::mem::zeroed() }; nb];
    quantize_row_q8_k_neon(&test_input, &mut x_q8k);

    println!("Input quantized to {} Q8_K blocks", nb);

    // Test single row GEMV
    let weights: &[BlockQ6K] = unsafe {
        std::slice::from_raw_parts(
            w2.bytes().as_ptr() as *const BlockQ6K,
            hidden_size * nb,
        )
    };

    println!("Weight has {} blocks", weights.len());

    // Compute first 5 rows
    for row in 0..5 {
        let result = unsafe {
            vec_dot_q6_k_q8_k_neon(
                intermediate_size,
                &weights[row * nb..(row + 1) * nb],
                &x_q8k,
            )
        };
        println!("Row {} result: {:.4}", row, result);
    }

    // Full GEMV
    let mut output = vec![0.0f32; hidden_size];
    for i in 0..hidden_size {
        output[i] = unsafe {
            vec_dot_q6_k_q8_k_neon(
                intermediate_size,
                &weights[i * nb..(i + 1) * nb],
                &x_q8k,
            )
        };
    }

    println!(
        "\nFull GEMV output (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        output[0], output[1], output[2], output[3], output[4]
    );

    let non_zero_count = output.iter().filter(|&&x| x != 0.0).count();
    println!("Non-zero values: {}/{}", non_zero_count, hidden_size);

    // Compare with gemv_impl
    let output2 = ctx.model.gemv_impl(&test_input, w2, hidden_size).unwrap();
    println!(
        "\ngemv_impl output (first 5): {:.4} {:.4} {:.4} {:.4} {:.4}",
        output2[0], output2[1], output2[2], output2[3], output2[4]
    );

    let non_zero_count2 = output2.iter().filter(|&&x| x != 0.0).count();
    println!("Non-zero values: {}/{}", non_zero_count2, hidden_size);
}
