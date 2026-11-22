// Test Q6_K vec_dot correctness
use infer_train::backend::arm::quant::quantize_q8_k;
use infer_train::backend::arm::vec_dot::vec_dot_q6_k_q8_k;
use infer_train::types::{BlockQ6_K, BlockQ8_K};

fn main() {
    let n = 256; // One block
    let x_f32: Vec<f32> =
        (0..n).map(|i| (i as f32 % 64.0 - 32.0) / 31.0).collect();

    // Quantize to Q8_K
    let mut q8_blocks =
        vec![BlockQ8_K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }; 1];
    quantize_q8_k(&x_f32, &mut q8_blocks);

    // Create a simple Q6_K block
    let mut q6_block = BlockQ6_K {
        ql: [0u8; 128],
        qh: [0u8; 64],
        scales: [1i8; 16], // All scales = 1
        d: 0,              // Will set FP16 scale
    };

    // Set d = 1.0 in FP16
    q6_block.d = infer_train::types::fp32_to_fp16(1.0);

    // Pack x_f32 into Q6_K format
    // Q6_K: 6-bit values in [-32, 31]
    // For simplicity, just set ql and qh to match the input
    for i in 0..256 {
        let val = (x_f32[i] * 31.0 + 32.0).round() as i32; // [-32, 31] -> [0, 63]
        let val = val.clamp(0, 63) as u8;
        let low4 = val & 0xF;
        let high2 = (val >> 4) & 0x3;

        // This is simplified - need proper packing per llama.cpp layout
    }

    // Test reference dot product
    let ref_dot: f32 = x_f32.iter().map(|&v| v * v).sum();
    println!("Reference dot product: {}", ref_dot);

    // For now, just check if the function compiles
    // unsafe {
    //     let result = vec_dot_q6_k_q8_k(std::slice::from_ref(&q6_block), &q8_blocks);
    //     println!("Q6_K x Q8_K result: {}", result);
    // }
}
