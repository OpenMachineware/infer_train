// Test interleaved GEMV for Q5_K

#[cfg(target_arch = "aarch64")]
fn main() {
    use infer_train::backend::arm::gemv::{gemv_q5_k, gemv_q5_k_x8_neon};
    use infer_train::backend::arm::repack::repack_q5_k_to_x8;
    use infer_train::types::{BlockQ5_K, BlockQ5_Kx8, BlockQ8_K, QK_K};

    // Test 1: Simple values
    println!("=== Test 1: Simple values ===");
    test_simple();

    // Test 2: With high bits
    println!("\n=== Test 2: With high bits ===");
    test_with_high_bits();

    // Test 3: With different scales
    println!("\n=== Test 3: With scales ===");
    test_with_scales();

    // Test 4: Multi-block test
    println!("\n=== Test 4: Multi-block test ===");
    test_multi_block();

    // Test 5: Full test
    println!("\n=== Test 5: Full test ===");
    test_full();
}

#[cfg(target_arch = "aarch64")]
fn test_simple() {
    use infer_train::backend::arm::gemv::{gemv_q5_k, gemv_q5_k_x8_neon};
    use infer_train::backend::arm::repack::repack_q5_k_to_x8;
    use infer_train::types::{BlockQ5_K, BlockQ5_Kx8, BlockQ8_K, QK_K};

    let n_blocks = 1;
    let in_features = n_blocks * QK_K;
    let out_features = 8;

    let mut weights: Vec<BlockQ5_K> =
        Vec::with_capacity(out_features * n_blocks);
    for row in 0..out_features {
        let mut block = BlockQ5_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [32u8; 12],
            qh: [0u8; 32],
            qs: [0u8; 128],
        };
        for i in 0..128 {
            block.qs[i] = (row * 10) as u8;
        }
        weights.push(block);
    }

    let mut input: Vec<BlockQ8_K> = Vec::with_capacity(n_blocks);
    input.push(BlockQ8_K { d: 1.0, qs: [1i8; 256], bsums: [16i16; 16] });

    let mut output_std = vec![0.0f32; out_features];
    unsafe {
        gemv_q5_k(
            &weights,
            &input,
            &mut output_std,
            out_features,
            in_features,
            1,
        );
    }

    let src_stride = n_blocks * std::mem::size_of::<BlockQ5_K>();
    let mut weights_packed: Vec<u8> =
        vec![
            0u8;
            (out_features / 8) * n_blocks * std::mem::size_of::<BlockQ5_Kx8>()
        ];
    let weights_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * std::mem::size_of::<BlockQ5_K>(),
        )
    };
    repack_q5_k_to_x8(
        weights_bytes,
        src_stride,
        &mut weights_packed,
        out_features,
        n_blocks,
    );

    let weights_x8: &[BlockQ5_Kx8] = unsafe {
        std::slice::from_raw_parts(
            weights_packed.as_ptr() as *const BlockQ5_Kx8,
            (out_features / 8) * n_blocks,
        )
    };

    let mut output_x8 = vec![0.0f32; out_features];
    unsafe {
        gemv_q5_k_x8_neon(
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

#[cfg(target_arch = "aarch64")]
fn test_with_high_bits() {
    use infer_train::backend::arm::gemv::{gemv_q5_k, gemv_q5_k_x8_neon};
    use infer_train::backend::arm::repack::repack_q5_k_to_x8;
    use infer_train::types::{BlockQ5_K, BlockQ5_Kx8, BlockQ8_K, QK_K};

    let n_blocks = 1;
    let in_features = n_blocks * QK_K;
    let out_features = 8;

    let mut weights: Vec<BlockQ5_K> =
        Vec::with_capacity(out_features * n_blocks);
    for row in 0..out_features {
        let mut block = BlockQ5_K {
            d: half::f16::from_f32(1.0).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [32u8; 12],
            qh: [0xFFu8; 32], // All high bits set
            qs: [0u8; 128],
        };
        for i in 0..128 {
            block.qs[i] = (row * 10) as u8;
        }
        weights.push(block);
    }

    let mut input: Vec<BlockQ8_K> = Vec::with_capacity(n_blocks);
    input.push(BlockQ8_K { d: 1.0, qs: [1i8; 256], bsums: [16i16; 16] });

    let mut output_std = vec![0.0f32; out_features];
    unsafe {
        gemv_q5_k(
            &weights,
            &input,
            &mut output_std,
            out_features,
            in_features,
            1,
        );
    }

    let src_stride = n_blocks * std::mem::size_of::<BlockQ5_K>();
    let mut weights_packed: Vec<u8> =
        vec![
            0u8;
            (out_features / 8) * n_blocks * std::mem::size_of::<BlockQ5_Kx8>()
        ];
    let weights_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * std::mem::size_of::<BlockQ5_K>(),
        )
    };
    repack_q5_k_to_x8(
        weights_bytes,
        src_stride,
        &mut weights_packed,
        out_features,
        n_blocks,
    );

    let weights_x8: &[BlockQ5_Kx8] = unsafe {
        std::slice::from_raw_parts(
            weights_packed.as_ptr() as *const BlockQ5_Kx8,
            (out_features / 8) * n_blocks,
        )
    };

    let mut output_x8 = vec![0.0f32; out_features];
    unsafe {
        gemv_q5_k_x8_neon(
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

#[cfg(target_arch = "aarch64")]
fn test_with_scales() {
    use infer_train::backend::arm::gemv::{gemv_q5_k, gemv_q5_k_x8_neon};
    use infer_train::backend::arm::repack::repack_q5_k_to_x8;
    use infer_train::types::{BlockQ5_K, BlockQ5_Kx8, BlockQ8_K, QK_K};

    let n_blocks = 1;
    let in_features = n_blocks * QK_K;
    let out_features = 8;

    let mut weights: Vec<BlockQ5_K> =
        Vec::with_capacity(out_features * n_blocks);
    for row in 0..out_features {
        let mut block = BlockQ5_K {
            d: half::f16::from_f32(0.1).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [(row + 1) as u8; 12], // Different scales per row
            qh: [0u8; 32],
            qs: [(row * 8) as u8; 128],
        };
        weights.push(block);
    }

    let mut input: Vec<BlockQ8_K> = Vec::with_capacity(n_blocks);
    input.push(BlockQ8_K { d: 0.5, qs: [1i8; 256], bsums: [16i16; 16] });

    let mut output_std = vec![0.0f32; out_features];
    unsafe {
        gemv_q5_k(
            &weights,
            &input,
            &mut output_std,
            out_features,
            in_features,
            1,
        );
    }

    let src_stride = n_blocks * std::mem::size_of::<BlockQ5_K>();
    let mut weights_packed: Vec<u8> =
        vec![
            0u8;
            (out_features / 8) * n_blocks * std::mem::size_of::<BlockQ5_Kx8>()
        ];
    let weights_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * std::mem::size_of::<BlockQ5_K>(),
        )
    };
    repack_q5_k_to_x8(
        weights_bytes,
        src_stride,
        &mut weights_packed,
        out_features,
        n_blocks,
    );

    let weights_x8: &[BlockQ5_Kx8] = unsafe {
        std::slice::from_raw_parts(
            weights_packed.as_ptr() as *const BlockQ5_Kx8,
            (out_features / 8) * n_blocks,
        )
    };

    let mut output_x8 = vec![0.0f32; out_features];
    unsafe {
        gemv_q5_k_x8_neon(
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

#[cfg(target_arch = "aarch64")]
fn test_multi_block() {
    use infer_train::backend::arm::gemv::{gemv_q5_k, gemv_q5_k_x8_neon};
    use infer_train::backend::arm::repack::repack_q5_k_to_x8;
    use infer_train::types::{BlockQ5_K, BlockQ5_Kx8, BlockQ8_K, QK_K};

    let n_blocks = 4;
    let in_features = n_blocks * QK_K;
    let out_features = 8;

    let mut weights: Vec<BlockQ5_K> =
        Vec::with_capacity(out_features * n_blocks);
    for row in 0..out_features {
        for blk in 0..n_blocks {
            let block = BlockQ5_K {
                d: half::f16::from_f32(1.0).to_bits(),
                dmin: half::f16::from_f32(0.0).to_bits(),
                scales: [32u8; 12],
                qh: [0u8; 32],
                qs: [((row * 10 + blk) % 256) as u8; 128],
            };
            weights.push(block);
        }
    }

    let mut input: Vec<BlockQ8_K> = Vec::with_capacity(n_blocks);
    for blk in 0..n_blocks {
        input.push(BlockQ8_K { d: 1.0, qs: [1i8; 256], bsums: [16i16; 16] });
    }

    let mut output_std = vec![0.0f32; out_features];
    unsafe {
        gemv_q5_k(
            &weights,
            &input,
            &mut output_std,
            out_features,
            in_features,
            1,
        );
    }

    let src_stride = n_blocks * std::mem::size_of::<BlockQ5_K>();
    let mut weights_packed: Vec<u8> =
        vec![
            0u8;
            (out_features / 8) * n_blocks * std::mem::size_of::<BlockQ5_Kx8>()
        ];
    let weights_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * std::mem::size_of::<BlockQ5_K>(),
        )
    };
    repack_q5_k_to_x8(
        weights_bytes,
        src_stride,
        &mut weights_packed,
        out_features,
        n_blocks,
    );

    let weights_x8: &[BlockQ5_Kx8] = unsafe {
        std::slice::from_raw_parts(
            weights_packed.as_ptr() as *const BlockQ5_Kx8,
            (out_features / 8) * n_blocks,
        )
    };

    let mut output_x8 = vec![0.0f32; out_features];
    unsafe {
        gemv_q5_k_x8_neon(
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

#[cfg(target_arch = "aarch64")]
fn test_full() {
    use infer_train::backend::arm::gemv::{gemv_q5_k, gemv_q5_k_x8_neon};
    use infer_train::backend::arm::repack::repack_q5_k_to_x8;
    use infer_train::types::{BlockQ5_K, BlockQ5_Kx8, BlockQ8_K, QK_K};

    let n_blocks = 4;
    let in_features = n_blocks * QK_K;
    let out_features = 16;

    let mut weights: Vec<BlockQ5_K> =
        Vec::with_capacity(out_features * n_blocks);
    for row in 0..out_features {
        for blk in 0..n_blocks {
            // Create properly encoded scales
            // scales[0..3] encode the low 6 bits of scales[0..3]
            // scales[4..7] encode the low 6 bits of mins[0..3]
            // scales[8..11] encode the high bits (set to 0 for simplicity)
            let mut scales = [0u8; 12];
            let scale_val = ((row + blk) % 64) as u8;
            let min_val = ((row + blk + 8) % 64) as u8;

            // Set scales[0..3]
            for i in 0..4 {
                scales[i] = scale_val;
            }
            // Set mins[0..3] via scales[4..7]
            for i in 4..8 {
                scales[i] = min_val;
            }
            // scales[8..11] stay as 0 (no high bits)

            let block = BlockQ5_K {
                d: half::f16::from_f32(0.1 + row as f32 * 0.01).to_bits(),
                dmin: half::f16::from_f32(0.01).to_bits(),
                scales,
                qh: [((row * 32 + blk * 32) % 256) as u8; 32],
                qs: [((row * 128 + blk * 128) % 256) as u8; 128],
            };
            weights.push(block);
        }
    }

    let mut input: Vec<BlockQ8_K> = Vec::with_capacity(n_blocks);
    for blk in 0..n_blocks {
        input.push(BlockQ8_K {
            d: 0.5,
            qs: [((blk * 64) % 128) as i8; 256],
            bsums: [(blk * 8) as i16; 16],
        });
    }

    let mut output_std = vec![0.0f32; out_features];
    unsafe {
        gemv_q5_k(
            &weights,
            &input,
            &mut output_std,
            out_features,
            in_features,
            1,
        );
    }

    let src_stride = n_blocks * std::mem::size_of::<BlockQ5_K>();
    let mut weights_packed: Vec<u8> =
        vec![
            0u8;
            (out_features / 8) * n_blocks * std::mem::size_of::<BlockQ5_Kx8>()
        ];
    let weights_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * std::mem::size_of::<BlockQ5_K>(),
        )
    };
    repack_q5_k_to_x8(
        weights_bytes,
        src_stride,
        &mut weights_packed,
        out_features,
        n_blocks,
    );

    let weights_x8: &[BlockQ5_Kx8] = unsafe {
        std::slice::from_raw_parts(
            weights_packed.as_ptr() as *const BlockQ5_Kx8,
            (out_features / 8) * n_blocks,
        )
    };

    let mut output_x8 = vec![0.0f32; out_features];
    unsafe {
        gemv_q5_k_x8_neon(
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

    if max_diff < 100.0 {
        println!("PASS");
    } else {
        println!("FAIL");
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    println!("Interleaved GEMV only supported on aarch64");
}
