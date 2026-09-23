// IQ2_XS multi-size benchmark
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use infer_train::quant::types::*;
use half::f16;

fn generate_iq2_xs_weights(m: usize, k: usize) -> Vec<BlockIQ2XS> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u16; 32];
        let mut scales = [0u8; 8];
        for j in 0..32 { qs[j] = ((i * 17 + j * 13) % 65536) as u16; }
        for j in 0..8 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockIQ2XS { d: f16::from_f32(1.0).to_bits(), qs, scales }
    }).collect()
}

fn generate_f32_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

fn main() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let k = 4096;
    let ms = [256, 512, 1024, 2048, 4096];
    let n_iter = 500;

    println!("=== Rust IQ2_XS Metal MV Multi-Size Benchmark ===");
    println!("K={}, N_ITER={}\n", k, n_iter);

    let input = generate_f32_input(k);

    println!("     M |   Time (µs) | GFLOPS");
    println!("-------|-------------|--------");

    for &m in &ms {
        let weights = generate_iq2_xs_weights(m, k);

        // Warmup
        let _ = ctx.mv_iq2_xs_f32(m, k, &weights, &input).expect("GPU failed");

        // Benchmark
        let start = std::time::Instant::now();
        for _ in 0..n_iter {
            let _ = ctx.mv_iq2_xs_f32(m, k, &weights, &input).expect("GPU failed");
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / n_iter as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (avg_us * 1e-6) / 1e9;

        println!("{:6} | {:11.2} | {:7.2}", m, avg_us, gflops);
    }
}
