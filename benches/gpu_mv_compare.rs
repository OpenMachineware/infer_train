// Compare MetalContext MV vs raw Metal API
use half::f16;
use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use std::time::Instant;

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
                scales[j] = ((i + j) % 64) as u8;
            }
            BlockQ4K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.0).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn main() {
    let m = 4096usize;
    let k = 4096usize;
    let iterations = 10;

    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let weights = generate_q4_k_weights(m, k);
    let input: Vec<f32> = (0..k).map(|i| (i as f32).sin()).collect();

    // Warmup
    let _ = ctx.mv_q4_k_f32(m, k, &weights, &input).expect("GPU failed");

    // Measure MetalContext
    let mut total_time = 0u128;
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = ctx.mv_q4_k_f32(m, k, &weights, &input).expect("GPU failed");
        total_time += start.elapsed().as_micros();
    }
    let avg_time = total_time as f64 / iterations as f64;
    let gflops = (m * k * 2) as f64 / (avg_time * 1e-6) / 1e9;
    println!("MetalContext avg over {} iterations:", iterations);
    println!("  Time: {:.2} us", avg_time);
    println!("  GFLOPS: {:.2}", gflops);

    // Now test with same buffers (no re-allocation)
    // MetalContext caches buffers, so 2nd call should be faster
    let mut total_time2 = 0u128;
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = ctx.mv_q4_k_f32(m, k, &weights, &input).expect("GPU failed");
        total_time2 += start.elapsed().as_micros();
    }
    let avg_time2 = total_time2 as f64 / iterations as f64;
    let gflops2 = (m * k * 2) as f64 / (avg_time2 * 1e-6) / 1e9;
    println!("\nMetalContext 2nd run (buffers cached):");
    println!("  Time: {:.2} us", avg_time2);
    println!("  GFLOPS: {:.2}", gflops2);
}
