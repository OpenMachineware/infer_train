// Test vec_dot performance directly
use infer_train::quant::vec_dot::arm::*;
use infer_train::quant::types::*;
use half::f16;
use std::time::Instant;

fn generate_q8_k_input(k: usize) -> Vec<BlockQ8K> {
    let nb = k / 256;
    (0..nb)
        .map(|i| {
            let mut qs = [0i8; 256];
            let mut bsums = [0i16; 16];
            for j in 0..256 {
                qs[j] = (((i * 17 + j * 13) % 256) as i32 - 128) as i8;
            }
            for j in 0..16 {
                bsums[j] = ((i * 23 + j * 7) % 256) as i16;
            }
            BlockQ8K {
                d: 0.01 + (i % 10) as f32 * 0.001,
                qs,
                bsums,
            }
        })
        .collect()
}

fn generate_q4_k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 23 + j * 7) % 256) as u8;
            }
            BlockQ4K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.1 + (i % 5) as f32 * 0.05).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn main() {
    unsafe {
        let k = 4096;
        let nb = k / 256;
        let x = generate_q8_k_input(k);
        let weights = generate_q4_k_weights(1, k);

        // Check slice sizes
        println!("x: {} blocks, weights: {} blocks, expected nb: {}", x.len(), weights.len(), nb);

        // Test vec_dot directly
        let start = Instant::now();
        for _ in 0..100000 {
            let _ = vec_dot_q4_k_q8_k_neon(k, &weights[0..nb], &x);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let gflops = 2.0 * (k as f64) * 100000.0 / elapsed / 1e9;

        println!("vec_dot Q4_K: {:.1} GFLOPS", gflops);
    }
}
