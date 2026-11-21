// Test tiled implementation correctness by comparing with vec_dot

use infer_train::backend::arm::gemv::mul_mat_q5_k;
use infer_train::backend::arm::tiled::{
    mul_mat_q5_k_tiled, mul_mat_q6_k_tiled,
};
use infer_train::backend::arm::vec_dot::*;
use infer_train::types::*;
use rand::Rng;

/// Create a random Q5_K block for testing
fn create_random_q5k_block() -> BlockQ5_K {
    let mut rng = rand::thread_rng();

    let d = rng.gen_range(0.1..1.0);
    let dmin = rng.gen_range(0.01..0.1);

    let mut scales = [0u8; 12];
    let mut qh = [0u8; 32];
    let mut qs = [0u8; 128];

    for i in 0..12 {
        scales[i] = rng.gen_range(0..64);
    }
    for i in 0..32 {
        qh[i] = rng.gen_range(0..256) as u8;
    }
    for i in 0..128 {
        qs[i] = rng.gen_range(0..256) as u8;
    }

    BlockQ5_K {
        d: half::f16::from_f32(d).to_bits(),
        dmin: half::f16::from_f32(dmin).to_bits(),
        scales,
        qh,
        qs,
    }
}

/// Create a random Q6_K block for testing
fn create_random_q6k_block() -> BlockQ6_K {
    let mut rng = rand::thread_rng();

    let d = rng.gen_range(0.1..1.0);

    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    let mut scales = [0i8; 16];

    for i in 0..128 {
        ql[i] = rng.gen_range(0..256) as u8;
    }
    for i in 0..64 {
        qh[i] = rng.gen_range(0..256) as u8;
    }
    for i in 0..16 {
        scales[i] = rng.gen_range(-64..64);
    }

    BlockQ6_K { d: half::f16::from_f32(d).to_bits(), ql, qh, scales }
}

/// Create a random Q8_K block for testing with correctly computed bsums
fn create_random_q8k_block() -> BlockQ8_K {
    let mut rng = rand::thread_rng();

    let d = rng.gen_range(0.1..1.0);
    let mut qs = [0i8; 256];
    let mut bsums = [0i16; 16];

    for i in 0..256 {
        qs[i] = rng.gen_range(-128..127);
    }

    // Compute bsums: each bsum is the sum of 16 consecutive qs values
    for i in 0..16 {
        let mut sum = 0i32;
        for j in 0..16 {
            sum += qs[i * 16 + j] as i32;
        }
        bsums[i] = sum.clamp(-32768, 32767) as i16;
    }

    BlockQ8_K { d, qs, bsums }
}

#[test]
fn test_tiled_q5_k_correctness() {
    // Small test case
    let out_features = 16; // Smaller for debugging
    let in_features = 256;
    let n_blocks = in_features / QK_K;
    let n_tokens = 1;

    // Create random weights
    let blocks_per_row = n_blocks;
    let mut weights_bytes = Vec::with_capacity(
        out_features * blocks_per_row * std::mem::size_of::<BlockQ5_K>(),
    );

    for _ in 0..out_features {
        for _ in 0..blocks_per_row {
            let block = create_random_q5k_block();
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &block as *const _ as *const u8,
                    std::mem::size_of::<BlockQ5_K>(),
                )
            };
            weights_bytes.extend_from_slice(bytes);
        }
    }

    // Create random activations
    let mut input = Vec::with_capacity(n_tokens * n_blocks);
    for _ in 0..n_tokens * n_blocks {
        input.push(create_random_q8k_block());
    }

    // Compute with original implementation
    let mut output_orig = vec![0.0f32; n_tokens * out_features];
    mul_mat_q5_k(
        &weights_bytes,
        blocks_per_row * std::mem::size_of::<BlockQ5_K>(),
        &input,
        &mut output_orig,
        out_features,
        in_features,
        n_tokens,
        1,
    );

    // Compute with tiled implementation
    let mut output_tiled = vec![0.0f32; n_tokens * out_features];
    mul_mat_q5_k_tiled(
        &weights_bytes,
        blocks_per_row * std::mem::size_of::<BlockQ5_K>(),
        &input,
        &mut output_tiled,
        out_features,
        in_features,
        n_tokens,
    );

    // Debug: print first few results
    println!("First 8 results comparison:");
    for i in 0..8.min(out_features) {
        println!(
            "  row {}: orig = {}, tiled = {}, diff = {}",
            i,
            output_orig[i],
            output_tiled[i],
            (output_orig[i] - output_tiled[i]).abs()
        );
    }

    // Compare results
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0;
    for i in 0..output_orig.len() {
        let diff = (output_orig[i] - output_tiled[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    println!("Max difference: {} at index {}", max_diff, max_diff_idx);

    // Allow small floating point differences
    assert!(
        max_diff < 1e-3,
        "Tiled implementation differs too much from original"
    );
}

#[test]
fn test_tiled_q6_k_correctness() {
    let out_features = 64;
    let in_features = 256;
    let n_blocks = in_features / QK_K;
    let n_tokens = 1;

    // Create random weights
    let blocks_per_row = n_blocks;
    let mut weights_bytes = Vec::with_capacity(
        out_features * blocks_per_row * std::mem::size_of::<BlockQ6_K>(),
    );

    for _ in 0..out_features {
        for _ in 0..blocks_per_row {
            let block = create_random_q6k_block();
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &block as *const _ as *const u8,
                    std::mem::size_of::<BlockQ6_K>(),
                )
            };
            weights_bytes.extend_from_slice(bytes);
        }
    }

    // Create random activations
    let mut input = Vec::with_capacity(n_tokens * n_blocks);
    for _ in 0..n_tokens * n_blocks {
        input.push(create_random_q8k_block());
    }

    // Compute with original implementation
    let mut output_orig = vec![0.0f32; n_tokens * out_features];
    infer_train::backend::arm::gemv::mul_mat_q6_k(
        &weights_bytes,
        blocks_per_row * std::mem::size_of::<BlockQ6_K>(),
        &input,
        &mut output_orig,
        out_features,
        in_features,
        n_tokens,
        1,
    );

    // Compute with tiled implementation
    let mut output_tiled = vec![0.0f32; n_tokens * out_features];
    mul_mat_q6_k_tiled(
        &weights_bytes,
        blocks_per_row * std::mem::size_of::<BlockQ6_K>(),
        &input,
        &mut output_tiled,
        out_features,
        in_features,
        n_tokens,
    );

    // Compare results
    let mut max_diff = 0.0f32;
    for i in 0..output_orig.len() {
        let diff = (output_orig[i] - output_tiled[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Max difference: {}", max_diff);

    assert!(
        max_diff < 1e-3,
        "Tiled implementation differs too much from original"
    );
}
