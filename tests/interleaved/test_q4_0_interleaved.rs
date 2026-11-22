use infer_train::backend::arm::gemv::{
    gemv_q4_0_transposed, gemv_q4_0_x8_neon,
};
use infer_train::backend::arm::repack::repack_q4_0_to_x8;
use infer_train::types::{BlockQ4_0, BlockQ4_0x8, BlockQ8_0, QK4_0};

fn main() {
    println!("=== Q4_0 Interleaved GEMV Test ===\n");

    // Test 1: Simple values
    // Q4_0 stores nibbles as (value + 8), so:
    // - Nibble 0 represents -8
    // - Nibble 8 represents 0
    // - Nibble F represents 7
    // Let's use nibble A (value 10 -> actual 2) and nibble 8 (value 8 -> actual 0)
    {
        let in_features = 32; // 1 block
        let out_features = 8; // 8 rows

        // Create Q4_0 weights: each byte has two nibbles of A (actual value 2)
        // A = 10, stored as nibble A, actual value = 10 - 8 = 2
        let weights: Vec<BlockQ4_0> = (0..out_features)
            .map(|_| BlockQ4_0 {
                d: half::f16::from_f32(1.0).to_bits(),
                qs: [0xAA; 16], // Each byte has two nibbles of A (actual value 2)
            })
            .collect();

        // Create Q8_0 input: all 1s
        let input =
            BlockQ8_0 { d: half::f16::from_f32(1.0).to_bits(), qs: [1i8; 32] };

        // Expected: 32 values of 2 * 1 = 64 per row
        let expected = 64.0f32;

        // Standard GEMV
        let mut output_std = vec![0.0f32; out_features];
        let stride = std::mem::size_of::<BlockQ4_0>();
        let weights_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                weights.as_ptr() as *const u8,
                out_features * std::mem::size_of::<BlockQ4_0>(),
            )
        };
        unsafe {
            gemv_q4_0_transposed(
                weights_bytes,
                stride,
                std::slice::from_ref(&input),
                &mut output_std,
                out_features,
                in_features,
                1,
            );
        }

        // Repack and interleaved GEMV
        let mut weights_packed: Vec<u8> =
            vec![0u8; std::mem::size_of::<BlockQ4_0x8>()];
        repack_q4_0_to_x8(
            weights_bytes,
            stride,
            &mut weights_packed,
            out_features,
            1,
        );
        let weights_x8: &BlockQ4_0x8 =
            unsafe { &*(weights_packed.as_ptr() as *const BlockQ4_0x8) };

        let mut output_x8 = vec![0.0f32; out_features];
        unsafe {
            gemv_q4_0_x8_neon(
                std::slice::from_ref(weights_x8),
                std::slice::from_ref(&input),
                &mut output_x8,
                out_features,
                in_features,
            );
        }

        println!("Test 1: Simple values (expected: {})", expected);
        println!("Standard: {:?}", &output_std[..4]);
        println!("Interleaved: {:?}", &output_x8[..4]);
        let max_diff = (0..out_features)
            .map(|i| (output_std[i] - output_x8[i]).abs())
            .fold(0.0f32, f32::max);
        println!("Max diff: {:.6}", max_diff);
        println!("{}", if max_diff < 0.01 { "PASS" } else { "FAIL" });
    }

    // Test 2: With scales
    {
        let in_features = 32;
        let out_features = 8;

        // Weights with different scales per row
        let weights: Vec<BlockQ4_0> = (0..out_features)
            .map(|i| BlockQ4_0 {
                d: half::f16::from_f32(1.0 + i as f32 * 0.1).to_bits(),
                qs: [0xAA; 16],
            })
            .collect();

        let input =
            BlockQ8_0 { d: half::f16::from_f32(1.0).to_bits(), qs: [1i8; 32] };

        let mut output_std = vec![0.0f32; out_features];
        let stride = std::mem::size_of::<BlockQ4_0>();
        let weights_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                weights.as_ptr() as *const u8,
                out_features * std::mem::size_of::<BlockQ4_0>(),
            )
        };
        unsafe {
            gemv_q4_0_transposed(
                weights_bytes,
                stride,
                std::slice::from_ref(&input),
                &mut output_std,
                out_features,
                in_features,
                1,
            );
        }

        let mut weights_packed: Vec<u8> =
            vec![0u8; std::mem::size_of::<BlockQ4_0x8>()];
        repack_q4_0_to_x8(
            weights_bytes,
            stride,
            &mut weights_packed,
            out_features,
            1,
        );
        let weights_x8: &BlockQ4_0x8 =
            unsafe { &*(weights_packed.as_ptr() as *const BlockQ4_0x8) };

        let mut output_x8 = vec![0.0f32; out_features];
        unsafe {
            gemv_q4_0_x8_neon(
                std::slice::from_ref(weights_x8),
                std::slice::from_ref(&input),
                &mut output_x8,
                out_features,
                in_features,
            );
        }

        println!("\nTest 2: With scales");
        println!("Standard: {:?}", &output_std[..4]);
        println!("Interleaved: {:?}", &output_x8[..4]);
        let max_diff = (0..out_features)
            .map(|i| (output_std[i] - output_x8[i]).abs())
            .fold(0.0f32, f32::max);
        println!("Max diff: {:.6}", max_diff);
        println!("{}", if max_diff < 0.01 { "PASS" } else { "FAIL" });
    }

    // Test 3: Multi-block
    {
        let n_blocks = 4;
        let in_features = 32 * n_blocks;
        let out_features = 8;

        let weights: Vec<BlockQ4_0> = (0..out_features * n_blocks)
            .map(|_| BlockQ4_0 {
                d: half::f16::from_f32(1.0).to_bits(),
                qs: [0xAA; 16],
            })
            .collect();

        let input: Vec<BlockQ8_0> = (0..n_blocks)
            .map(|_| BlockQ8_0 {
                d: half::f16::from_f32(1.0).to_bits(),
                qs: [1i8; 32],
            })
            .collect();

        let mut output_std = vec![0.0f32; out_features];
        let stride = n_blocks * std::mem::size_of::<BlockQ4_0>();
        let weights_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                weights.as_ptr() as *const u8,
                weights.len() * std::mem::size_of::<BlockQ4_0>(),
            )
        };
        unsafe {
            gemv_q4_0_transposed(
                weights_bytes,
                stride,
                &input,
                &mut output_std,
                out_features,
                in_features,
                1,
            );
        }

        let mut weights_packed: Vec<u8> =
            vec![0u8; std::mem::size_of::<BlockQ4_0x8>() * n_blocks];
        repack_q4_0_to_x8(
            weights_bytes,
            stride,
            &mut weights_packed,
            out_features,
            n_blocks,
        );
        let weights_x8: &[BlockQ4_0x8] = unsafe {
            std::slice::from_raw_parts(
                weights_packed.as_ptr() as *const BlockQ4_0x8,
                n_blocks,
            )
        };

        let mut output_x8 = vec![0.0f32; out_features];
        unsafe {
            gemv_q4_0_x8_neon(
                weights_x8,
                &input,
                &mut output_x8,
                out_features,
                in_features,
            );
        }

        println!(
            "\nTest 3: Multi-block (expected: {})",
            64.0 * n_blocks as f32
        );
        println!("Standard: {:?}", &output_std[..4]);
        println!("Interleaved: {:?}", &output_x8[..4]);
        let max_diff = (0..out_features)
            .map(|i| (output_std[i] - output_x8[i]).abs())
            .fold(0.0f32, f32::max);
        println!("Max diff: {:.6}", max_diff);
        println!("{}", if max_diff < 0.01 { "PASS" } else { "FAIL" });
    }
}
