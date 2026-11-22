// Test interleaved GEMV for Q4_K
use infer_train::backend::arm::gemv::{
    gemv_q4_k_transposed, gemv_q4_k_x8_neon,
};
use infer_train::backend::arm::repack::repack_q4_k_to_x8;
use infer_train::types::{BlockQ4_K, BlockQ4_Kx8, BlockQ8_K, QK_K};

fn main() {
    println!("=== Q4_K Interleaved GEMV Test ===\n");

    // Test 1: Simple values
    println!("Test 1: Simple values");
    test_simple();

    // Test 2: With scales
    println!("\nTest 2: With scales");
    test_with_scales();

    // Test 3: Multi-block
    println!("\nTest 3: Multi-block");
    test_multi_block();
}

fn test_simple() {
    let n_blocks = 1;
    let in_features = n_blocks * QK_K;
    let out_features = 8;

    // Create weights with simple scales
    let mut weights: Vec<BlockQ4_K> =
        Vec::with_capacity(out_features * n_blocks);
    for row in 0..out_features {
        let mut scales = [0u8; 12];
        for i in 0..4 {
            scales[i] = 32; // scales[0..3] = 32
        }
        for i in 4..8 {
            scales[i] = 16; // mins[0..3] = 16
        }

        let block = BlockQ4_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales,
            qs: [128u8; 128], // All nibbles = 8
        };
        weights.push(block);
    }

    // Create input
    let mut input: Vec<BlockQ8_K> = Vec::with_capacity(n_blocks);
    input.push(BlockQ8_K { d: 1.0, qs: [1i8; 256], bsums: [16i16; 16] });

    // Standard GEMV
    let mut output_std = vec![0.0f32; out_features];
    let stride = n_blocks * std::mem::size_of::<BlockQ4_K>();
    let weights_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * std::mem::size_of::<BlockQ4_K>(),
        )
    };
    unsafe {
        gemv_q4_k_transposed(
            weights_bytes,
            stride,
            &input,
            &mut output_std,
            out_features,
            in_features,
            1,
        );
    }

    // Repack and interleaved GEMV
    let mut weights_packed: Vec<u8> =
        vec![
            0u8;
            (out_features / 8) * n_blocks * std::mem::size_of::<BlockQ4_Kx8>()
        ];
    repack_q4_k_to_x8(
        weights_bytes,
        stride,
        &mut weights_packed,
        out_features,
        n_blocks,
    );

    let weights_x8: &[BlockQ4_Kx8] = unsafe {
        std::slice::from_raw_parts(
            weights_packed.as_ptr() as *const BlockQ4_Kx8,
            (out_features / 8) * n_blocks,
        )
    };

    let mut output_x8 = vec![0.0f32; out_features];
    unsafe {
        gemv_q4_k_x8_neon(
            weights_x8,
            &input,
            &mut output_x8,
            out_features,
            in_features,
        );
    }

    println!("Standard: {:?}", &output_std);
    println!("Interleaved: {:?}", &output_x8);

    let max_diff = (0..out_features)
        .map(|i| (output_std[i] - output_x8[i]).abs())
        .fold(0.0, f32::max);
    println!("Max diff: {:.6}", max_diff);

    if max_diff < 1.0 {
        println!("PASS");
    } else {
        println!("FAIL");
    }
}

fn test_with_scales() {
    let n_blocks = 1;
    let in_features = n_blocks * QK_K;
    let out_features = 8;

    let mut weights: Vec<BlockQ4_K> =
        Vec::with_capacity(out_features * n_blocks);
    for row in 0..out_features {
        let mut scales = [0u8; 12];
        // Set different scales for each row
        for i in 0..4 {
            scales[i] = 32 + row as u8; // scales vary by row
        }
        for i in 4..8 {
            scales[i] = 16;
        }

        let block = BlockQ4_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales,
            qs: [128u8; 128],
        };
        weights.push(block);
    }

    let mut input: Vec<BlockQ8_K> = Vec::with_capacity(n_blocks);
    input.push(BlockQ8_K { d: 1.0, qs: [1i8; 256], bsums: [16i16; 16] });

    let mut output_std = vec![0.0f32; out_features];
    let stride = n_blocks * std::mem::size_of::<BlockQ4_K>();
    let weights_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * std::mem::size_of::<BlockQ4_K>(),
        )
    };
    unsafe {
        gemv_q4_k_transposed(
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
        vec![
            0u8;
            (out_features / 8) * n_blocks * std::mem::size_of::<BlockQ4_Kx8>()
        ];
    repack_q4_k_to_x8(
        weights_bytes,
        stride,
        &mut weights_packed,
        out_features,
        n_blocks,
    );

    let weights_x8: &[BlockQ4_Kx8] = unsafe {
        std::slice::from_raw_parts(
            weights_packed.as_ptr() as *const BlockQ4_Kx8,
            (out_features / 8) * n_blocks,
        )
    };

    let mut output_x8 = vec![0.0f32; out_features];
    unsafe {
        gemv_q4_k_x8_neon(
            weights_x8,
            &input,
            &mut output_x8,
            out_features,
            in_features,
        );
    }

    println!("Standard: {:?}", &output_std);
    println!("Interleaved: {:?}", &output_x8);

    let max_diff = (0..out_features)
        .map(|i| (output_std[i] - output_x8[i]).abs())
        .fold(0.0, f32::max);
    println!("Max diff: {:.6}", max_diff);

    if max_diff < 1.0 {
        println!("PASS");
    } else {
        println!("FAIL");
    }
}

fn test_multi_block() {
    let n_blocks = 4;
    let in_features = n_blocks * QK_K;
    let out_features = 8;

    let mut weights: Vec<BlockQ4_K> =
        Vec::with_capacity(out_features * n_blocks);
    for row in 0..out_features {
        for blk in 0..n_blocks {
            let mut scales = [0u8; 12];
            for i in 0..4 {
                scales[i] = 32;
            }
            for i in 4..8 {
                scales[i] = 16;
            }

            let block = BlockQ4_K {
                d: half::f16::from_f32(1.0).to_bits(),
                dmin: half::f16::from_f32(0.5).to_bits(),
                scales,
                qs: [128u8; 128],
            };
            weights.push(block);
        }
    }

    let mut input: Vec<BlockQ8_K> = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        input.push(BlockQ8_K { d: 1.0, qs: [1i8; 256], bsums: [16i16; 16] });
    }

    let mut output_std = vec![0.0f32; out_features];
    let stride = n_blocks * std::mem::size_of::<BlockQ4_K>();
    let weights_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * std::mem::size_of::<BlockQ4_K>(),
        )
    };
    unsafe {
        gemv_q4_k_transposed(
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
        vec![
            0u8;
            (out_features / 8) * n_blocks * std::mem::size_of::<BlockQ4_Kx8>()
        ];
    repack_q4_k_to_x8(
        weights_bytes,
        stride,
        &mut weights_packed,
        out_features,
        n_blocks,
    );

    let weights_x8: &[BlockQ4_Kx8] = unsafe {
        std::slice::from_raw_parts(
            weights_packed.as_ptr() as *const BlockQ4_Kx8,
            (out_features / 8) * n_blocks,
        )
    };

    let mut output_x8 = vec![0.0f32; out_features];
    unsafe {
        gemv_q4_k_x8_neon(
            weights_x8,
            &input,
            &mut output_x8,
            out_features,
            in_features,
        );
    }

    println!("Standard: {:?}", &output_std);
    println!("Interleaved: {:?}", &output_x8);

    let max_diff = (0..out_features)
        .map(|i| (output_std[i] - output_x8[i]).abs())
        .fold(0.0, f32::max);
    println!("Max diff: {:.6}", max_diff);

    if max_diff < 1.0 {
        println!("PASS");
    } else {
        println!("FAIL");
    }
}
