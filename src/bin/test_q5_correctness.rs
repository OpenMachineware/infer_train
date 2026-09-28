// Test Q5_0 and Q5_1 vec_dot correctness
use infer_train::quant::types::{BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockQ8_1, QK4_0};
use infer_train::quant::vec_dot::arm::{vec_dot_q5_0_q8_0_neon, vec_dot_q5_1_q8_1_neon};
use half::f16;

fn main() {
    unsafe {
        println!("Testing Q5_1 correctness (fixed version with TABLE_B2B_0 + vorrq_s8)");

        // Q5_1: 5-bit unsigned, range 0-31
        // value = low4bit | (high_bit << 4)

        let mut x = BlockQ5_1 {
            d: f16::from_f32(1.0).to_bits(),
            m: f16::from_f32(0.0).to_bits(),
            qh: [0u8; 4],
            qs: [0u8; 16],
        };

        // Set bit 0 (element 0) = 1, all others = 0
        x.qh[0] = 0b00000001;

        // Element 0: low4bit = 0, high_bit = 1 → value = 0 | 16 = 16
        // Element 1-15: low4bit = 0, high_bit = 0 → value = 0 | 0 = 0
        x.qs[0] = 0;

        let mut y = BlockQ8_1 {
            d: f16::from_f32(1.0).to_bits(),
            s: f16::from_f32(0.0).to_bits(),
            qs: [1i8; 32],
        };

        let result = vec_dot_q5_1_q8_1_neon(QK4_0, &[x.clone()], &[y.clone()]);
        // Expected: element 0 contributes 16*1, others contribute 0*1
        // But we have 32 elements in y, so:
        // qh has 32 bits for 32 elements
        // Bits 0-15 are for low nibbles, bits 16-31 are for high nibbles
        // We set bit 0, so element 0's low nibble gets high bit = 1
        // Value = 0 | 16 = 16
        // All other elements: high bit = 0, value = 0
        // So sum = 16 + 0 + 0 + ... = 16
        println!("Q5_1 test 1: result = {}, expected ~16 (element 0 = 16, rest = 0)", result);

        // Test with value 17 (binary 10001)
        x.qh[0] = 0b00000001; // bit 0 = 1
        x.qs[0] = 0x01; // low nibble = 1, high nibble = 0
        // Element 0: low4bit = 1, high_bit = 1 → value = 1 | 16 = 17
        // Element 1: low4bit = 0, high_bit = 0 → value = 0
        // ...
        // Element 16: high nibble of qs[0] = 0, bit 16 of qh = 0 → value = 0

        let result2 = vec_dot_q5_1_q8_1_neon(QK4_0, &[x.clone()], &[y.clone()]);
        println!("Q5_1 test 2: result = {}, expected ~17 (element 0 = 17, rest = 0)", result2);

        println!("\nTesting Q5_0 correctness");
        // Q5_0: 5-bit signed, range -16 to 15
        // value = low4bit - table[bit] where table: bit=0→0x10, bit=1→0x00

        let mut x0 = BlockQ5_0 {
            d: f16::from_f32(1.0).to_bits(),
            qh: [0u8; 4],
            qs: [0u8; 16],
        };

        let mut y0 = BlockQ8_0 {
            d: f16::from_f32(1.0).to_bits(),
            qs: [1i8; 32],
        };

        // Test with all bits = 1 (should give low4bit - 0 = low4bit)
        x0.qh = [0xFF, 0xFF, 0xFF, 0xFF]; // all 32 bits = 1
        x0.qs[0] = 0x88; // low nibble = 8, high nibble = 8
        // Element 0: bit = 1, low4bit = 8, result = 8 - 0 = 8
        // Element 16: bit = 1, high nibble = 8, result = 8 - 0 = 8

        let result3 = vec_dot_q5_0_q8_0_neon(QK4_0, &[x0.clone()], &[y0.clone()]);
        println!("Q5_0 test 1: result = {}, all bits=1, low nibbles=8", result3);
        // Expected: each element = 8, 32 elements, sum = 256

        // Test with all bits = 0 (should give low4bit - 16)
        x0.qh = [0u8; 4]; // all 32 bits = 0
        x0.qs[0] = 0x00; // low nibble = 0, high nibble = 0
        // Element 0: bit = 0, low4bit = 0, result = 0 - 16 = -16
        // Element 16: bit = 0, high nibble = 0, result = 0 - 16 = -16

        let result4 = vec_dot_q5_0_q8_0_neon(QK4_0, &[x0.clone()], &[y0.clone()]);
        println!("Q5_0 test 2: result = {}, all bits=0, low nibbles=0", result4);
        // Expected: each element = -16, 32 elements, sum = -512
    }
}
