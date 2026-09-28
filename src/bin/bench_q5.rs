// Benchmark Q5_0 and Q5_1 vec_dot performance
use half::f16;
use infer_train::quant::types::{
    BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockQ8_1, QK4_0,
};
use infer_train::quant::vec_dot::arm::{
    vec_dot_q5_0_q8_0_neon, vec_dot_q5_1_q8_1_neon,
};
use std::time::Instant;

/// Warm up CPU to stabilize frequency
fn warmup(iterations: usize) {
    let mut sum = 0.0f64;
    for i in 0..iterations {
        sum = sum.sqrt() + i as f64;
    }
    // Prevent compiler from optimizing away
    if sum < 0.0 {
        println!("This should never print");
    }
}

fn generate_q8_0_input(k: usize) -> Vec<BlockQ8_0> {
    let nb = k / 32;
    (0..nb)
        .map(|i| {
            let mut qs = [0i8; 32];
            for j in 0..32 {
                qs[j] = (((i * 17 + j * 13) % 256) as i32 - 128) as i8;
            }
            BlockQ8_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_q8_1_input(k: usize) -> Vec<BlockQ8_1> {
    let nb = k / 32;
    (0..nb)
        .map(|i| {
            let mut qs = [0i8; 32];
            for j in 0..32 {
                qs[j] = (((i * 17 + j * 13) % 256) as i32 - 128) as i8;
            }
            BlockQ8_1 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                s: f16::from_f32((i % 5) as f32 * 0.2).to_bits(),
                qs,
            }
        })
        .collect()
}

fn generate_q5_0_weights(k: usize) -> Vec<BlockQ5_0> {
    let nb = k / 32;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            let mut qh = [0u8; 4];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..4 {
                qh[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            BlockQ5_0 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qh,
                qs,
            }
        })
        .collect()
}

fn generate_q5_1_weights(k: usize) -> Vec<BlockQ5_1> {
    let nb = k / 32;
    (0..nb)
        .map(|i| {
            let mut qs = [0u8; 16];
            let mut qh = [0u8; 4];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..4 {
                qh[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            BlockQ5_1 {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                m: f16::from_f32((i % 5) as f32 * 0.2).to_bits(),
                qh,
                qs,
            }
        })
        .collect()
}

fn main() {
    // Warm up CPU to stabilize frequency
    println!("Warming up CPU...");
    warmup(50_000_000); // More iterations for better warmup
    std::thread::sleep(std::time::Duration::from_millis(500)); // Longer sleep

    unsafe {
        let k = 4096;
        let iterations = 10000;

        println!("Q5 Vec Dot Performance Benchmark (K={})", k);
        println!("{}", "=".repeat(50));

        // Q5_0
        let weights_q5_0 = generate_q5_0_weights(k);
        let input_q8_0 = generate_q8_0_input(k);
        let mut dst = [0.0f32; 1];

        let start = Instant::now();
        for _ in 0..iterations {
            dst[0] = vec_dot_q5_0_q8_0_neon(k, &weights_q5_0, &input_q8_0);
        }
        let elapsed = start.elapsed();
        let ops = iterations as f64 * 2.0 * k as f64;
        let gflops_q5_0 = ops / elapsed.as_secs_f64() / 1e9;

        println!(
            "Q5_0: {:.2} GFLOPS ({:.2} ms for {} iterations)",
            gflops_q5_0,
            elapsed.as_secs_f64() * 1000.0,
            iterations
        );

        // Q5_1
        let weights_q5_1 = generate_q5_1_weights(k);
        let input_q8_1 = generate_q8_1_input(k);

        let start = Instant::now();
        for _ in 0..iterations {
            dst[0] = vec_dot_q5_1_q8_1_neon(k, &weights_q5_1, &input_q8_1);
        }
        let elapsed = start.elapsed();
        let ops = iterations as f64 * 2.0 * k as f64;
        let gflops_q5_1 = ops / elapsed.as_secs_f64() / 1e9;

        println!(
            "Q5_1: {:.2} GFLOPS ({:.2} ms for {} iterations)",
            gflops_q5_1,
            elapsed.as_secs_f64() * 1000.0,
            iterations
        );

        println!("{}", "=".repeat(50));

        // llama.cpp baseline from memory (K=4096):
        // Q5_0: 20.98 GFLOPS
        // Q5_1: 18.61 GFLOPS
        println!("\nllama.cpp baseline (from memory):");
        println!("Q5_0: 20.98 GFLOPS");
        println!("Q5_1: 18.61 GFLOPS");

        println!("\nComparison:");
        println!("Q5_0: {:.1}% of llama.cpp", gflops_q5_0 / 20.98 * 100.0);
        println!("Q5_1: {:.1}% of llama.cpp", gflops_q5_1 / 18.61 * 100.0);
    }
}
