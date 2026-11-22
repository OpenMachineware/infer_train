use infer_train::backend::arm::gemv::gemv_q6_k_x8_neon;
use infer_train::backend::arm::repack::repack_q6_k_to_x8;
use infer_train::backend::arm::vec_dot::vec_dot_q6_k_q8_k;
use infer_train::types::{BlockQ6_K, BlockQ6_Kx8, BlockQ8_K};

fn main() {
    println!("=== Q6_K Interleaved GEMV Test ===\n");

    // Test 1: Single block, 8 rows
    {
        let in_features = 256; // 1 superblock
        let out_features = 8; // 8 rows

        // Create Q6_K weights
        let weights: Vec<BlockQ6_K> = (0..out_features)
            .map(|i| BlockQ6_K {
                ql: [((i % 16) + 1) as u8; 128],
                qh: [((i % 4) + 1) as u8; 64],
                scales: [1 + (i % 4) as i8; 16],
                d: half::f16::from_f32(1.0 + (i % 10) as f32 * 0.1).to_bits(),
            })
            .collect();

        // Create Q8_K input
        let input = BlockQ8_K { d: 1.0f32, qs: [1i8; 256], bsums: [16i16; 16] };

        // Scalar reference
        let mut output_scalar = vec![0.0f32; out_features];
        for i in 0..out_features {
            output_scalar[i] = vec_dot_q6_k_q8_k(
                &weights[i..i + 1],
                std::slice::from_ref(&input),
            );
        }

        // Repack weights
        let stride = std::mem::size_of::<BlockQ6_K>();
        let weights_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                weights.as_ptr() as *const u8,
                out_features * std::mem::size_of::<BlockQ6_K>(),
            )
        };

        let mut weights_packed: Vec<u8> =
            vec![0u8; std::mem::size_of::<BlockQ6_Kx8>()];
        repack_q6_k_to_x8(
            weights_bytes,
            stride,
            &mut weights_packed,
            out_features,
            1,
        );
        let weights_x8: &BlockQ6_Kx8 =
            unsafe { &*(weights_packed.as_ptr() as *const BlockQ6_Kx8) };

        // Interleaved GEMV
        let mut output_interleaved = vec![0.0f32; out_features];
        unsafe {
            gemv_q6_k_x8_neon(
                std::slice::from_ref(weights_x8),
                std::slice::from_ref(&input),
                &mut output_interleaved,
                out_features,
                in_features,
            );
        }

        println!("Test 1: Single block, 8 rows");
        println!("Scalar:     {:?}", &output_scalar[..4]);
        println!("Interleaved: {:?}", &output_interleaved[..4]);

        let max_diff = (0..out_features)
            .map(|i| (output_scalar[i] - output_interleaved[i]).abs())
            .fold(0.0f32, f32::max);
        println!("Max diff: {:.6}", max_diff);
        println!("{}", if max_diff < 1.0 { "PASS" } else { "FAIL" });
    }

    // Test 2: Multi-block
    {
        println!("\n--- Test 2: Multi-block ---");
        let n_blocks = 4;
        let in_features = 256 * n_blocks;
        let out_features = 8;

        let weights: Vec<BlockQ6_K> = (0..out_features * n_blocks)
            .map(|i| BlockQ6_K {
                ql: [((i % 16) + 1) as u8; 128],
                qh: [((i % 4) + 1) as u8; 64],
                scales: [1 + (i % 4) as i8; 16],
                d: half::f16::from_f32(1.0).to_bits(),
            })
            .collect();

        let input: Vec<BlockQ8_K> = (0..n_blocks)
            .map(|_| BlockQ8_K {
                d: 1.0f32,
                qs: [1i8; 256],
                bsums: [16i16; 16],
            })
            .collect();

        // Scalar reference
        let mut output_scalar = vec![0.0f32; out_features];
        for i in 0..out_features {
            let row_blocks: Vec<BlockQ6_K> =
                (0..n_blocks).map(|j| weights[i * n_blocks + j]).collect();
            output_scalar[i] = vec_dot_q6_k_q8_k(&row_blocks, &input);
        }

        // Repack
        let stride = n_blocks * std::mem::size_of::<BlockQ6_K>();
        let weights_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                weights.as_ptr() as *const u8,
                weights.len() * std::mem::size_of::<BlockQ6_K>(),
            )
        };

        let mut weights_packed: Vec<u8> =
            vec![0u8; n_blocks * std::mem::size_of::<BlockQ6_Kx8>()];
        repack_q6_k_to_x8(
            weights_bytes,
            stride,
            &mut weights_packed,
            out_features,
            n_blocks,
        );
        let weights_x8: &[BlockQ6_Kx8] = unsafe {
            std::slice::from_raw_parts(
                weights_packed.as_ptr() as *const BlockQ6_Kx8,
                n_blocks,
            )
        };

        // Interleaved GEMV
        let mut output_interleaved = vec![0.0f32; out_features];
        unsafe {
            gemv_q6_k_x8_neon(
                weights_x8,
                &input,
                &mut output_interleaved,
                out_features,
                in_features,
            );
        }

        println!("Scalar:     {:?}", &output_scalar[..4]);
        println!("Interleaved: {:?}", &output_interleaved[..4]);

        let max_diff = (0..out_features)
            .map(|i| (output_scalar[i] - output_interleaved[i]).abs())
            .fold(0.0f32, f32::max);
        println!("Max diff: {:.6}", max_diff);
        println!("{}", if max_diff < 1.0 { "PASS" } else { "FAIL" });
    }
}
