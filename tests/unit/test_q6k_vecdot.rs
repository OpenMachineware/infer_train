// Test Q6_K vec_dot correctness
use infer_train::backend::arm::quant::quantize_q8_k;
use infer_train::backend::arm::vec_dot::vec_dot_q6_k_q8_k;
use infer_train::types::{BlockQ6_K, BlockQ8_K, fp16_to_fp32, fp32_to_fp16};

fn pack_q6_k_value(q6: u8) -> (u8, u8, u8) {
    // Q6_K packing:
    // ql[i] holds low 4 bits of q6[i] in its low nibble
    // and low 4 bits of q6[i+128] in its high nibble
    // qh[i/2] holds high 2 bits for q6[i] and q6[i+1]
    //
    // Actually, let's check the llama.cpp layout more carefully.
    // From ggml-common.h:
    // uint8_t ql[QK_K/2];  // 128 bytes, low 4 bits for 256 values
    // uint8_t qh[QK_K/4];  // 64 bytes, high 2 bits for 256 values
    //
    // ql[i] holds low 4 bits for values i and i+128 (2 values per byte)
    // Actually no, that doesn't add up: 128 bytes * 2 values/byte = 256 values, but
    // each value needs 4 bits, so 128 bytes = 256 * 4 bits / 8 = 128 values
    //
    // Let me re-read the llama.cpp code for vec_dot_q6_k_q8_k.
    // The actual packing is:
    // For each super-block of 256 values:
    // - ql has 128 bytes, each byte holds low 4 bits of 2 consecutive values
    // - qh has 64 bytes, each byte holds high 2 bits of 4 consecutive values
    //
    // From vec_dot code:
    // q6 = (ql & 0x0F) | ((qh >> shift) & 0x03) << 4
    //
    // This seems to be:
    // - ql[i] & 0x0F gives low 4 bits
    // - qh bits give high 2 bits

    // For simplicity, let's just test the decoding
    (0, 0, 0)
}

fn main() {
    println!("=== Testing Q6_K vec_dot ===\n");

    // Test 1: Simple test with known values
    // Create a Q6_K block where all values are 32 (so q6-32 = 0)
    // and a Q8_K block with all values = 1
    // Result should be 0

    let mut q6k = BlockQ6_K {
        ql: [0u8; 128],
        qh: [0u8; 64],
        scales: [1i8; 16],
        d: fp32_to_fp16(1.0),
    };

    // Set ql and qh so q6 = 32 for all values
    // q6 = (ql & 0x0F) | ((qh >> shift) & 0x03) << 4
    // For q6 = 32 = 0b100000:
    // - low 4 bits = 0
    // - high 2 bits = 2
    // So ql[i] & 0x0F = 0, and ((qh >> shift) & 0x03) = 2
    //
    // From llama.cpp vec_dot:
    // q6h_0 = (qh_bits.0 & 0x03) << 4  // first 16 values
    // q6h_1 = (qh_bits.1 & 0x03) << 4  // next 16 values
    // etc.
    //
    // q6 = (ql & 0x0F) | q6h
    //
    // So for high bits = 2: we need qh_bits & 0x03 = 2
    // For all q6 = 32: high 2 bits = 2, low 4 bits = 0

    // Set ql low nibbles to 0 (already done by zero init)
    // Set qh to provide high 2 bits = 2
    // qh[0] bits 0-1 for value 0, bits 2-3 for value 1, etc.
    for i in 0..64 {
        // Each qh byte holds 4 x 2-bit values
        // For value 2 in each position: 2 | (2 << 2) | (2 << 4) | (2 << 6)
        q6k.qh[i] = 0b10101010; // = 0xAA = 2 | 8 | 32 | 128
    }

    // Q8_K block with all values = 1
    let mut q8k = BlockQ8_K {
        d: 1.0f32,
        qs: [1i8; 256],
        bsums: [16i16; 16], // Each block of 16 has sum = 16
    };

    // Expected result:
    // q6 values = 32, so q6-32 = 0
    // sum = 0 * 256 = 0
    let result = unsafe { vec_dot_q6_k_q8_k(&[q6k], &[q8k.clone()]) };
    println!("Test 1: All q6=32, all q8=1");
    println!("  Result: {}", result);
    println!("  Expected: 0.0");
    println!("  Status: {}", if result.abs() < 0.01 { "PASS" } else { "FAIL" });

    // Test 2: All q6 = 0 (so q6-32 = -32)
    let mut q6k_zero = BlockQ6_K {
        ql: [0u8; 128],
        qh: [0u8; 64],
        scales: [1i8; 16],
        d: fp32_to_fp16(1.0),
    };

    let result_zero = unsafe { vec_dot_q6_k_q8_k(&[q6k_zero], &[q8k.clone()]) };
    println!("\nTest 2: All q6=0 (so q6-32=-32), all q8=1");
    println!("  Result: {}", result_zero);
    println!("  Expected: {} (256 * -32 * 1 * 1)", 256.0 * (-32.0));
    println!(
        "  Status: {}",
        if (result_zero - (-8192.0)).abs() < 1.0 { "PASS" } else { "FAIL" }
    );

    // Test 3: All q6 = 63 (so q6-32 = 31)
    let mut q6k_max = BlockQ6_K {
        ql: [0u8; 128],
        qh: [0u8; 64],
        scales: [1i8; 16],
        d: fp32_to_fp16(1.0),
    };

    // For q6 = 63 = 0b111111:
    // low 4 bits = 15 = 0xF
    // high 2 bits = 3
    //
    // ql[i] low nibble = 15: ql[i] = (something << 4) | 0xF
    // From code: ql_bits.0, ql_bits.1, etc.
    // ql & 0x0F gives low 4 bits, ql >> 4 gives next value's low 4 bits
    // So for all values having low 4 bits = 15:
    // ql[i] = 0xFF (both nibbles = 15)
    for i in 0..128 {
        q6k_max.ql[i] = 0xFF;
    }

    // For high 2 bits = 3:
    // qh_bits & 0x03 = 3, so qh = 3 | (3 << 2) | (3 << 4) | (3 << 6) = 0xFF
    for i in 0..64 {
        q6k_max.qh[i] = 0xFF;
    }

    let result_max = unsafe { vec_dot_q6_k_q8_k(&[q6k_max], &[q8k.clone()]) };
    println!("\nTest 3: All q6=63 (so q6-32=31), all q8=1");
    println!("  Result: {}", result_max);
    println!("  Expected: {} (256 * 31 * 1 * 1)", 256.0 * 31.0);
    println!(
        "  Status: {}",
        if (result_max - 7936.0).abs() < 1.0 { "PASS" } else { "FAIL" }
    );

    // Test 4: Check scale handling
    let mut q6k_scaled = BlockQ6_K {
        ql: [0u8; 128],
        qh: [0u8; 64],
        scales: [2i8; 16], // All scales = 2
        d: fp32_to_fp16(1.0),
    };

    let result_scaled =
        unsafe { vec_dot_q6_k_q8_k(&[q6k_scaled], &[q8k.clone()]) };
    println!("\nTest 4: All q6=0, all scales=2, all q8=1");
    println!("  Result: {}", result_scaled);
    println!("  Expected: {} (256 * -32 * 2 * 1)", 256.0 * (-32.0) * 2.0);
    println!(
        "  Status: {}",
        if (result_scaled - (-16384.0)).abs() < 1.0 { "PASS" } else { "FAIL" }
    );

    // Test 5: Check d (super-block scale)
    let mut q6k_d = BlockQ6_K {
        ql: [0u8; 128],
        qh: [0u8; 64],
        scales: [1i8; 16],
        d: fp32_to_fp16(2.0), // d = 2
    };

    let result_d = unsafe { vec_dot_q6_k_q8_k(&[q6k_d], &[q8k]) };
    println!("\nTest 5: All q6=0, all scales=1, d=2, all q8=1");
    println!("  Result: {}", result_d);
    println!("  Expected: {} (256 * -32 * 1 * 2)", 256.0 * (-32.0) * 2.0);
    println!(
        "  Status: {}",
        if (result_d - (-16384.0)).abs() < 1.0 { "PASS" } else { "FAIL" }
    );
}
