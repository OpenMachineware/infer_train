// Correctness test: Q4_0 and Q5_1 vs llama.cpp

use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::arm::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

unsafe extern "C" {
    fn llama_vec_dot_q4_0_q8_0(
        n: i32,
        s: *mut f32,
        x: *const BlockQ4_0,
        y: *const BlockQ8_0,
    );
    fn llama_vec_dot_q5_1_q8_1(
        n: i32,
        s: *mut f32,
        x: *const BlockQ5_1,
        y: *const BlockQ8_1,
    );
}

fn main() {
    let n_blocks = 256;
    let n = n_blocks * QK8_0;

    let mut rng = StdRng::seed_from_u64(42);

    // Test Q4_0
    {
        let mut x_blocks: Vec<BlockQ4_0> = Vec::with_capacity(n_blocks);
        let mut y_blocks: Vec<BlockQ8_0> = Vec::with_capacity(n_blocks);

        for _ in 0..n_blocks {
            let mut qs = [0u8; QK8_0 / 2];
            let mut q8 = [0i8; QK8_0];
            for j in 0..QK8_0 / 2 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..QK8_0 {
                q8[j] = (rng.r#gen::<u8>() as i8) - 64;
            }
            let d = 1.0f32 + rng.r#gen::<f32>() * 2.0;

            x_blocks
                .push(BlockQ4_0 { d: half::f16::from_f32(d).to_bits(), qs });
            y_blocks.push(BlockQ8_0 {
                d: half::f16::from_f32(d).to_bits(),
                qs: q8,
            });
        }

        let mut llama_result = 0.0f32;
        unsafe {
            llama_vec_dot_q4_0_q8_0(
                n as i32,
                &mut llama_result,
                x_blocks.as_ptr(),
                y_blocks.as_ptr(),
            );
        }

        let rust_result =
            unsafe { vec_dot_q4_0_q8_0_neon(n, &x_blocks, &y_blocks) };

        println!("Q4_0:");
        println!("  llama.cpp: {}", llama_result);
        println!("  Rust:      {}", rust_result);
        println!(
            "  Diff:      {:.6}%",
            (llama_result - rust_result).abs() / llama_result.abs() * 100.0
        );
    }

    // Test Q5_1
    {
        let mut x_blocks: Vec<BlockQ5_1> = Vec::with_capacity(n_blocks);
        let mut y_blocks: Vec<BlockQ8_1> = Vec::with_capacity(n_blocks);

        for _ in 0..n_blocks {
            let mut qs = [0u8; QK8_0 / 2];
            let mut qh = [0u8; 4];
            let mut q8 = [0i8; QK8_0];
            for j in 0..QK8_0 / 2 {
                qs[j] = rng.r#gen::<u8>();
            }
            for j in 0..4 {
                qh[j] = rng.r#gen::<u8>();
            }
            for j in 0..QK8_0 {
                q8[j] = (rng.r#gen::<u8>() as i8) - 64;
            }
            let d = 1.0f32 + rng.r#gen::<f32>() * 2.0;
            let m = rng.r#gen::<f32>() * 2.0;
            let s = rng.r#gen::<f32>() * 100.0;

            x_blocks.push(BlockQ5_1 {
                d: half::f16::from_f32(d).to_bits(),
                m: half::f16::from_f32(m).to_bits(),
                qs,
                qh,
            });
            y_blocks.push(BlockQ8_1 {
                d: half::f16::from_f32(d).to_bits(),
                s: half::f16::from_f32(s).to_bits(),
                qs: q8,
            });
        }

        let mut llama_result = 0.0f32;
        unsafe {
            llama_vec_dot_q5_1_q8_1(
                n as i32,
                &mut llama_result,
                x_blocks.as_ptr(),
                y_blocks.as_ptr(),
            );
        }

        let rust_result =
            unsafe { vec_dot_q5_1_q8_1_neon(n, &x_blocks, &y_blocks) };

        println!("Q5_1:");
        println!("  llama.cpp: {}", llama_result);
        println!("  Rust:      {}", rust_result);
        println!(
            "  Diff:      {:.6}%",
            (llama_result - rust_result).abs() / llama_result.abs() * 100.0
        );
    }
}
