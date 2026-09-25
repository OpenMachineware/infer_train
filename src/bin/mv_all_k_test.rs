// Comprehensive GPU MV vs llama.cpp verification
// Test all formats across all K sizes

use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use infer_train::quant::types::*;
use half::f16;
use std::time::Instant;

fn generate_f32_vector(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect()
}

fn generate_q4_0_weights(m: usize, k: usize) -> Vec<BlockQ4_0> {
    let nb = k / 32;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 16];
        for j in 0..16 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        BlockQ4_0 {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
        }
    }).collect()
}

fn generate_q4_k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 128];
        let mut scales = [0u8; 12];
        for j in 0..128 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..12 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockQ4K {
            d: f16::from_f32(0.5).to_bits(),
            dmin: f16::from_f32(0.1).to_bits(),
            scales,
            qs,
        }
    }).collect()
}

fn main() {
    let metal_ctx = MetalContext::new().expect("Failed to create Metal context");

    println!("GPU MV Performance - All Formats, All K Sizes");
    println!("================================================\n");

    let k_sizes = [256, 512, 1024, 2048, 4096];
    let m_sizes = [256, 1024, 4096];

    // Test Q4_0
    println!("## Q4_0 MV Performance\n");
    println!("K\\M\t256\t1024\t4096");

    for k in k_sizes.iter() {
        if k % 32 != 0 { continue; }
        print!("{}\t", k);

        for m in m_sizes.iter() {
            let weights = generate_q4_0_weights(*m, *k);
            let input = generate_f32_vector(*k);

            let start = Instant::now();
            for _ in 0..10 {
                let _ = metal_ctx.mv_q4_0_f32(*m, *k, &weights, &input);
            }
            let elapsed = start.elapsed().as_micros() as f64 / 10.0;
            let flops = 2.0 * *m as f64 * *k as f64;
            let gflops = flops / elapsed / 1000.0;

            print!("{:.1}\t", gflops);
        }
        println!();
    }

    // Test Q4_K
    println!("\n## Q4_K MV Performance\n");
    println!("K\\M\t256\t1024\t4096");

    for k in k_sizes.iter() {
        if k % 256 != 0 { continue; }
        print!("{}\t", k);

        for m in m_sizes.iter() {
            let weights = generate_q4_k_weights(*m, *k);
            let input = generate_f32_vector(*k);

            let start = Instant::now();
            for _ in 0..10 {
                let _ = metal_ctx.mv_q4_k_f32(*m, *k, &weights, &input);
            }
            let elapsed = start.elapsed().as_micros() as f64 / 10.0;
            let flops = 2.0 * *m as f64 * *k as f64;
            let gflops = flops / elapsed / 1000.0;

            print!("{:.1}\t", gflops);
        }
        println!();
    }

    println!("\n## Next Steps");
    println!("1. Run llama-bench with Q4_0 and Q4_K models");
    println!("2. Compare MV performance at each K size");
    println!("3. Verify we exceed llama.cpp for all cases");
}
