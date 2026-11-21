//! Test Q6_K vec_dot implementation correctness

use infer_train::types::{BlockQ6_K, BlockQ8_K};
use infer_train::backend::arm::vec_dot::vec_dot_q6_k_q8_k;
use std::mem::size_of;

fn main() {
    // Create a simple test case
    // Q6_K block: 256 elements in 210 bytes
    // Q8_K block: 256 elements in 292 bytes

    let mut q6_block = BlockQ6_K {
        ql: [0u8; 128],
        qh: [0u8; 64],
        scales: [1i8; 16],  // All scales = 1
        d: 15360u16, // FP16 for 1.0
    };

    let mut q8_block = BlockQ8_K {
        d: 1.0f32,
        qs: [1i8; 256],  // All q8 values = 1
        bsums: [16i16; 16], // Each 16-element block sums to 16
    };

    // Set ql values to represent 32 (so after -32, we get 0)
    // q6 value 32 = 0b100000 = low 4 bits = 0, high 2 bits = 2
    for i in 0..128 {
        q6_block.ql[i] = 0; // Low 4 bits = 0
    }
    for i in 0..64 {
        q6_block.qh[i] = 0b00101010; // High 2 bits for positions 0,2,4,6 = 2
    }

    // Expected: (32 - 32) * 1 * 1 * 256 = 0

    let result = unsafe { vec_dot_q6_k_q8_k(&[q6_block], &[q8_block]) };
    println!("Test 1 - Expected ~0.0, got {}", result);

    // Test 2: q6 = 33 (after -32 = 1), q8 = 1
    // Expected: 1 * 1 * 256 = 256
    for i in 0..64 {
        q6_block.qh[i] = 0b00101010; // High bits = 2
    }
    for i in 0..32 {
        q6_block.ql[i] = 1; // Low bits = 1
    }
    // q6 = 1 + (2 << 4) = 33, after -32 = 1

    let result = unsafe { vec_dot_q6_k_q8_k(&[q6_block], &[q8_block]) };
    println!("Test 2 - Expected ~256.0, got {}", result);

    // Test 3: Use negative scale
    q6_block.scales[0] = -1;
    q6_block.scales[4] = -1;
    q6_block.scales[8] = -1;
    q6_block.scales[12] = -1;
    // Expected: 256 with some negative scales

    let result = unsafe { vec_dot_q6_k_q8_k(&[q6_block], &[q8_block]) };
    println!("Test 3 - With negative scales: {}", result);

    println!("\nQ6_K block size: {} bytes", size_of::<BlockQ6_K>());
    println!("Q8_K block size: {} bytes", size_of::<BlockQ8_K>());
}
