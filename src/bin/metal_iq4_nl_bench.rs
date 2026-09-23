// Benchmark IQ4_NL Metal kernel across multiple sizes
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use infer_train::quant::types::*;
use half::f16;

fn generate_f32_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

fn generate_iq4_nl_weights(m: usize, k: usize) -> Vec<BlockIQ4NL> {
    let nb = k / 32;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 16];
        for j in 0..16 {
            qs[j] = ((i * 17 + j * 13) % 256) as u8;
        }
        BlockIQ4NL { d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(), qs }
    }).collect()
}

fn main() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let k = 4096;
    let sizes = vec![256, 512, 1024, 2048, 4096];
    let n_iter = 500;

    println!("=== IQ4_NL Metal MV Benchmark ===");
    println!("K={}, N_ITER={}\n", k, n_iter);

    let input = generate_f32_input(k);

    for m in sizes {
        let weights = generate_iq4_nl_weights(m, k);

        // Warmup
        let _ = ctx.mv_iq4_nl_f32(m, k, &weights, &input).expect("GPU failed");

        let start = std::time::Instant::now();
        for _ in 0..n_iter {
            let _ = ctx.mv_iq4_nl_f32(m, k, &weights, &input).expect("GPU failed");
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / n_iter as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (avg_us * 1e-6) / 1e9;

        println!("M={:5}: {:8.2} µs, {:6.2} GFLOPS", m, avg_us, gflops);
    }
}
