// Test Q6_K vec_dot correctness
use infer_train::backend::arm::vec_dot::vec_dot_q6_k_q8_k;
use infer_train::types::{BlockQ6_K, BlockQ8_K};

fn create_test_blocks() -> (BlockQ6_K, BlockQ8_K) {
    // Create a simple Q6_K block where:
    // - All ql values are 0x20 (low 4 bits = 0)
    // - All qh values are 0 (high 2 bits = 0)
    // So all q6 values are 0, which means q6_unpacked - 32 = -32

    let mut q6k = BlockQ6_K {
        ql: [0u8; 128],
        qh: [0u8; 64],
        scales: [1i8; 16], // All scales = 1
        d: half::f16::from_f32(1.0).to_bits(),
    };

    // Set ql so q6 = 32 (which gives q6 - 32 = 0)
    // q6 = ql_low4 | (qh << 4)
    // For q6 = 32, we need ql_low4 = 0, qh = 2
    // But let's just set ql to 32 (low 4 bits = 0, high 4 bits = 2)
    // Actually, ql holds low 4 bits of q6
    // q6 = ql[i] & 0x0F | (qh bits) << 4
    // For q6 = 32 = 0b100000, we need low 4 bits = 0, high 2 bits = 2
    // So ql = 0, qh = 2 in the right position

    // Simpler: let q6 values be 32 (so q6-32=0)
    // ql[i] & 0x0F = 0, and qh[i/2] >> ((i%2)*2) & 0x3 = 2
    // So qh[i/2] = 2 << ((i%2)*2)
    for i in 0..64 {
        q6k.qh[i] =
            (2 << ((i % 2) * 2)) as u8 | ((2 << (((i + 1) % 2) * 2)) as u8);
    }

    // Q8_K block with all 1s
    let q8k = BlockQ8_K {
        d: 1.0f32,
        qs: [1i8; 256],
        bsums: [16i16; 16], // Each block of 16 has sum = 16 (16 * 1)
    };

    (q6k, q8k)
}

fn main() {
    let (q6k, q8k) = create_test_blocks();

    // Expected:
    // q6 values = 32, so q6-32 = 0
    // scale = 1, d = 1
    // q8 values = 1
    // So sum = 0 * 1 * 1 * 1 = 0

    let result = unsafe { vec_dot_q6_k_q8_k(&[q6k], &[q8k]) };
    println!("Result: {}", result);
    println!("Expected: 0.0");

    // Now test with q6 = 0 (so q6-32 = -32)
    let mut q6k_neg = BlockQ6_K {
        ql: [0u8; 128],
        qh: [0u8; 64],
        scales: [1i8; 16],
        d: half::f16::from_f32(1.0).to_bits(),
    };

    // q6 = 0 means ql = 0, qh = 0
    // q6-32 = -32
    // sum = -32 * 256 * 1 * 1 = -8192

    let result_neg = unsafe { vec_dot_q6_k_q8_k(&[q6k_neg], &[q8k]) };
    println!("\nResult with q6=0: {}", result_neg);
    println!("Expected: -8192.0 (256 values * -32 * 1)");

    // Test with q6 = 63 (max value)
    let mut q6k_max = BlockQ6_K {
        ql: [0u8; 128],
        qh: [0u8; 64],
        scales: [1i8; 16],
        d: half::f16::from_f32(1.0).to_bits(),
    };

    // q6 = 63 = 0b111111
    // low 4 bits = 15 = 0xF
    // high 2 bits = 3
    // So ql & 0x0F = 15, and we need to set ql accordingly
    // And qh needs to provide the high 2 bits = 3

    // For all ql, set low nibble to 15
    for i in 0..128 {
        q6k_max.ql[i] = 0xF0 | 0x0F; // Both nibbles = 15
    }
    // For all qh, set bits to 3
    // qh stores high 2 bits for each q6 value
    // qh[i/2] >> ((i%2)*2) & 0x3 gives the high 2 bits
    // We want this to be 3 for all i
    // So qh[i] = (3 << 0) | (3 << 2) | (3 << 4) | (3 << 6) = 0xFF
    for i in 0..64 {
        q6k_max.qh[i] = 0xFF;
    }

    let result_max = unsafe { vec_dot_q6_k_q8_k(&[q6k_max], &[q8k]) };
    println!("\nResult with q6=63: {}", result_max);
    println!(
        "Expected: {} (256 values * (63-32) * 1 = 7936)",
        256 * (63 - 32) as i32
    );
}
