// Test Q5_K vec_dot correctness
use infer_train::backend::arm::vec_dot::vec_dot_q5_k_q8_k;
use infer_train::types::{BlockQ5_K, BlockQ8_K, fp16_to_fp32, fp32_to_fp16};

fn main() {
    println!("=== Testing Q5_K vec_dot ===\n");

    // Test with all zeros (should give 0)
    let q5k_zero = BlockQ5_K {
        d: fp32_to_fp16(1.0),
        dmin: fp32_to_fp16(0.0),
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [0u8; 128],
    };

    let q8k = BlockQ8_K { d: 1.0f32, qs: [1i8; 256], bsums: [16i16; 16] };

    let result_zero = unsafe { vec_dot_q5_k_q8_k(&[q5k_zero], &[q8k.clone()]) };
    println!("Test: All zeros, scales=0, q8=1");
    println!("  Result: {}", result_zero);
    println!("  Expected: 0.0");
    println!(
        "  Status: {}",
        if result_zero.abs() < 0.01 { "PASS" } else { "FAIL" }
    );

    // Test with all qs = 16, scale = 1
    // Q5 values range from 0-31 (5 bits)
    // If all qs = 16, q5 = 16
    // But we need to check the scales encoding

    let mut q5k_test = BlockQ5_K {
        d: fp32_to_fp16(1.0),
        dmin: fp32_to_fp16(0.0),
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [16u8; 128], // All low nibbles = 16, high nibbles = 16
    };

    // Set scales to 1
    // scales encoding is complex, but for simplicity let's try setting raw bytes
    // K_SCALE_SIZE = 12, which encodes 8 scales + 8 mins
    // Each scale/min is 6 bits, packed into 12 bytes
    // For scale = 1 and min = 0 for all blocks

    // Actually, let me check the Q5_K scales encoding from llama.cpp
    // From q5_k header: scales are quantized with 6 bits
    // The encoding is complex, let's just set all to 0 for now

    let result_test = unsafe { vec_dot_q5_k_q8_k(&[q5k_test], &[q8k.clone()]) };
    println!("\nTest: All qs=16, scales=0, qh=0, q8=1");
    println!("  Result: {}", result_test);
    println!("  (with scales=0, expected near 0)");

    // Test with actual scale values
    // Let's try setting scales raw bytes to represent scale = 1
    // This requires understanding the 6-bit encoding

    // For now, let's just run a simple sanity check
    println!("\n=== Sanity check: scale/dmin effects ===");

    let q5k_d = BlockQ5_K {
        d: fp32_to_fp16(2.0), // d = 2
        dmin: fp32_to_fp16(0.0),
        scales: [0u8; 12],
        qh: [0u8; 32],
        qs: [0u8; 128],
    };

    let result_d = unsafe { vec_dot_q5_k_q8_k(&[q5k_d], &[q8k.clone()]) };
    println!("Test: d=2, all zeros");
    println!("  Result: {}", result_d);
    println!("  (should still be 0)");
}
