// Test single block vec_dot comparison

use half::f16;
use infer_train::backend::arm::tiled::{
    TiledTileSrc0, TiledTileSrc1, tiled_compute_q5_k, tiled_unpack_src0_q5_k,
    tiled_unpack_src1_q8_k,
};
use infer_train::backend::arm::vec_dot::vec_dot_q5_k_q8_k;
use infer_train::types::*;
use rand::Rng;

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
        d: f16::from_f32(d).to_bits(),
        dmin: f16::from_f32(dmin).to_bits(),
        scales,
        qh,
        qs,
    }
}

fn create_random_q8k_block() -> BlockQ8_K {
    let mut rng = rand::thread_rng();

    let d = rng.gen_range(0.1..1.0);
    let mut qs = [0i8; 256];
    let mut bsums = [0i16; 16];

    for i in 0..256 {
        qs[i] = rng.gen_range(-128..127);
    }

    // Compute bsums correctly
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
fn test_single_block_vec_dot() {
    // Create a single block pair
    let q5k_block = create_random_q5k_block();
    let q8k_block = create_random_q8k_block();

    // Compute with original vec_dot
    let orig_result = vec_dot_q5_k_q8_k(&[q5k_block], &[q8k_block]);

    // Compute with tiled approach
    // 1. Unpack Q5_K
    let mut src0_tile = TiledTileSrc0::new(1, 1, 8);
    let weight_bytes = unsafe {
        std::slice::from_raw_parts(
            &q5k_block as *const _ as *const u8,
            std::mem::size_of::<BlockQ5_K>(),
        )
    };
    tiled_unpack_src0_q5_k(
        weight_bytes,
        std::mem::size_of::<BlockQ5_K>(),
        1,
        &mut src0_tile,
        1,
        0,
        1,
    );

    // 2. Unpack Q8_K
    let mut src1_tile = TiledTileSrc1::new(1, 1);
    tiled_unpack_src1_q8_k(&[q8k_block], 1, 1, &mut src1_tile, 0, 1);

    // 3. Compute
    let mut acc = [0.0f32; 1];
    tiled_compute_q5_k(&src0_tile, &src1_tile, 1, 1, 1, &mut acc);

    let tiled_result = acc[0];

    println!("Orig result: {}", orig_result);
    println!("Tiled result: {}", tiled_result);
    println!("Difference: {}", (orig_result - tiled_result).abs());

    assert!(
        (orig_result - tiled_result).abs() < 1e-3,
        "Single block results differ too much"
    );
}
