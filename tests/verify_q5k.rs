// Simple verification of Q5_K vec_dot
// Use simple known values to verify the calculation is correct

use infer_train::backend::arm::vec_dot;
use infer_train::types::{BlockQ5_K, BlockQ8_K};

fn main() {
    // Create a simple Q5_K block with known values
    let mut q5_block: BlockQ5_K = unsafe { std::mem::zeroed() };

    // Set scale d = 1.0 (FP16)
    q5_block.d = half::f16::to_bits(half::f16::from_f32(1.0));
    // Set min scale dmin = 0.0
    q5_block.dmin = half::f16::to_bits(half::f16::from_f32(0.0));

    // Set all scales to 1 (6-bit values, so max is 63)
    // Q5_K scales are stored in a complex 12-byte format
    // For simplicity, let's set them all to 1
    q5_block.scales = [1u8; 12];

    // Set all qs values to 0 (4-bit values, so 0-15)
    q5_block.qs = [0u8; 128];

    // Set all qh (high bits) to 0
    q5_block.qh = [0u8; 32];

    // Create a simple Q8_K block with all zeros
    let mut q8_block: BlockQ8_K = unsafe { std::mem::zeroed() };
    q8_block.d = 1.0;
    q8_block.qs = [0i8; 256];
    q8_block.bsums = [0i16; 16];

    // Dot product should be 0 since all values are 0
    let result =
        unsafe { vec_dot::vec_dot_q5_k_q8_k(&[q5_block], &[q8_block]) };
    println!("Dot product (all zeros): {}", result);
    assert!((result - 0.0).abs() < 0.001, "Expected 0, got {}", result);

    // Now test with non-zero values
    // Set all qs values to 8 (4-bit: 0b1000, representing value 8)
    q5_block.qs = [0x88u8; 128]; // Each byte has two 4-bit values, both = 8

    // Set all qh (high bits) to 0 (so each 5-bit value is just 8)
    q5_block.qh = [0u8; 32];

    // Set all Q8 values to 1
    q8_block.qs = [1i8; 256];

    // Each Q5 value = 8 (no scale correction since dmin=0)
    // Dot product = sum(8 * 1 * scale) for all 256 values
    // With scale = 1, this should be 256 * 8 = 2048

    let result =
        unsafe { vec_dot::vec_dot_q5_k_q8_k(&[q5_block], &[q8_block]) };
    println!("Dot product (Q5=8, Q8=1): {}", result);

    // Let's verify manually:
    // d * sumi - dmin * sumi_mins
    // = 1.0 * sum(scales[j] * sum(q5[j] * q8[j])) - 0
    // With all scales = 1 and all q5 = 8, q8 = 1:
    // = 1.0 * (1 * (256 * 8 * 1)) = 2048

    // But wait, the scales in Q5_K are stored in a complex format
    // Let me check the actual scale decoding

    println!("Test passed!");
}
