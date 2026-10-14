// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Correctness test: compare Rust vs llama.cpp for IQ3_XXS

use infer_train::quant::types::{BlockIQ3XXS, BlockQ8K, QK_K};
use infer_train::quant::vec_dot::arm::vec_dot_iq3_xxs_q8_k_neon;
use rand::{Rng, SeedableRng, rngs::StdRng};

unsafe extern "C" {
    fn llama_vec_dot_iq3_xxs_q8_K(
        n: i32,
        s: *mut f32,
        x: *const BlockIQ3XXS,
        y: *const BlockQ8K,
    );
}

fn main() {
    let n_blocks = 4;
    let n = n_blocks * QK_K;

    // Create test data
    let mut rng = StdRng::seed_from_u64(42);
    let mut x_blocks = Vec::with_capacity(n_blocks);
    let mut y_blocks = Vec::with_capacity(n_blocks);

    for _ in 0..n_blocks {
        let mut qs = [0u8; 96];
        for byte in qs.iter_mut() {
            *byte = rng.r#gen::<u8>();
        }
        let d_f32 = 1.0f32 + rng.r#gen::<f32>() * 2.0;
        let d = d_f32.to_bits() as u16;

        x_blocks.push(BlockIQ3XXS { d, qs });
    }

    for _ in 0..n_blocks {
        let mut qs = [0i8; QK_K];
        for byte in qs.iter_mut() {
            *byte = (rng.r#gen::<u8>() as i8) - 64;
        }
        let d = 1.0f32 + rng.r#gen::<f32>() * 2.0;
        let bsums = [0i16; 16];

        y_blocks.push(BlockQ8K { d, qs, bsums });
    }

    let x: &[BlockIQ3XXS] = &x_blocks;
    let y: &[BlockQ8K] = &y_blocks;

    // llama.cpp result
    let mut llama_result = 0.0f32;
    unsafe {
        llama_vec_dot_iq3_xxs_q8_K(
            n as i32,
            &mut llama_result,
            x.as_ptr(),
            y.as_ptr(),
        );
    }

    // Rust result
    let rust_result = unsafe { vec_dot_iq3_xxs_q8_k_neon(n, x, y) };

    println!("llama.cpp result: {}", llama_result);
    println!("Rust result:      {}", rust_result);
    println!("Difference:       {}", (llama_result - rust_result).abs());
    println!(
        "Relative error:   {:.6}%",
        (llama_result - rust_result).abs() / llama_result.abs() * 100.0
    );
}
