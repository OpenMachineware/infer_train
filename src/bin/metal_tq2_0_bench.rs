// Test TQ2_0 Metal MV kernel
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use infer_train::quant::types::BlockTQ2_0;
use half::f16;

fn generate_tq2_0_weights(m: usize, k: usize) -> Vec<BlockTQ2_0> {
    let nb = k / 256;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..m * nb {
        let mut qs = [0u8; 64];
        for j in 0..64 {
            qs[j] = (i * 17 + j * 13) as u8;
        }
        weights.push(BlockTQ2_0 {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
        });
    }
    weights
}

fn generate_f32_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

fn main() {
    let m = 1024;
    let k = 4096;

    println!("Testing TQ2_0 Metal MV: M={}, K={}", m, k);

    let ctx = MetalContext::new().expect("Failed to init Metal");

    let weights = generate_tq2_0_weights(m, k);
    let input = generate_f32_input(k);

    // Warmup
    let output = ctx.mv_tq2_0_f32(m, k, &weights, &input).expect("MV failed");
    println!("Sample output[0]: {}", output[0]);

    // Benchmark
    let n_iter = 100;
    let start = std::time::Instant::now();
    for _ in 0..n_iter {
        let _ = ctx.mv_tq2_0_f32(m, k, &weights, &input).expect("MV failed");
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / n_iter as f64;

    println!("TQ2_0 Metal MV {}x{}: {:.2} µs/iter", m, k, avg_us);
    println!("GFLOPS: {:.2}", (2.0 * m as f64 * k as f64) / (avg_us * 1e-6) / 1e9);
}
