// Verify matmul correctness
use infer_train::quant::matmul::*;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::arm::*;
use half::f16;

fn main() {
    unsafe {
        println!("=== Matmul Correctness Test ===\n");

        // Test with K=256 (single block)
        let k = 256;
        let m = 4;

        // Create simple test data
        let x: Vec<BlockQ8K> = vec![BlockQ8K {
            d: 1.0,
            qs: [1i8; 256],  // All 1s
            bsums: [256i16; 16],  // Sum of 256 1s = 256
        }];

        // Create weights: each row has scale 1.0 and all quants as 8 (representing value 8)
        let weights: Vec<BlockQ4K> = (0..m).map(|_| {
            let mut qs = [0u8; 128];
            for j in 0..128 {
                qs[j] = 0x88;  // Both low and high nibble = 8
            }
            BlockQ4K {
                d: f16::from_f32(1.0).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales: [1u8; 12],  // All scales = 1
                qs,
            }
        }).collect();

        // Expected: each row * x = (8 * 1) * 256 * 1 = 2048
        // With scale 1.0 and d=1.0: result should be 2048

        let mut dst = vec![0.0f32; m];
        matmul_q4_k_q8_k(&weights, &x, &mut dst, k, m);

        println!("Q4_K matmul results:");
        for i in 0..m {
            println!("  dst[{}] = {:.2}", i, dst[i]);
        }

        // Compare with individual vec_dot calls
        println!("\nVec_dot individual results:");
        for i in 0..m {
            let result = vec_dot_q4_k_q8_k_neon(k, &weights[i..i+1], &x);
            println!("  vec_dot[{}] = {:.2}", i, result);
        }

        // Test with multiple K blocks
        println!("\n--- Testing K=1024 (4 blocks) ---");
        let k = 1024;
        let x: Vec<BlockQ8K> = (0..4).map(|_| BlockQ8K {
            d: 1.0,
            qs: [1i8; 256],
            bsums: [256i16; 16],
        }).collect();

        let weights: Vec<BlockQ4K> = (0..4).map(|_| {
            let mut qs = [0u8; 128];
            for j in 0..128 { qs[j] = 0x88; }
            BlockQ4K {
                d: f16::from_f32(1.0).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales: [1u8; 12],
                qs,
            }
        }).collect();

        let mut dst = [0.0f32; 1];
        matmul_q4_k_q8_k(&weights, &x, &mut dst, k, 1);

        let vec_dot_result = vec_dot_q4_k_q8_k_neon(k, &weights, &x);

        println!("matmul result: {:.2}", dst[0]);
        println!("vec_dot result: {:.2}", vec_dot_result);
        println!("Match: {}", (dst[0] - vec_dot_result).abs() < 0.001);
    }
}
