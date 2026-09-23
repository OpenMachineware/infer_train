use infer_train::quant::types::BlockIQ4NL;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

#[test]
fn bench_metal_iq4_nl_scaling() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    let k = 4096;
    let nb = k / 32;  // QK4_NL = 32

    for &m in &[64, 128, 256, 512, 1024, 2048, 4096, 8192] {
        let mut x_blocks = Vec::with_capacity(m * nb);
        let mut input = vec![0.0f32; k];

        for i in 0..m * nb {
            let mut qs = [0u8; 16];
            for j in 0..16 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            x_blocks.push(BlockIQ4NL {
                d: half::f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                qs,
            });
        }

        for i in 0..k {
            input[i] = (i % 10) as f32 * 0.1;
        }

        let _ = ctx.mv_iq4_nl_f32(m, k, &x_blocks, &input).expect("GPU failed");

        let n_iter = 100;
        let start = std::time::Instant::now();

        for _ in 0..n_iter {
            let _ = ctx.mv_iq4_nl_f32(m, k, &x_blocks, &input).expect("GPU failed");
        }

        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / n_iter as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (avg_us * 1e-6) / 1e9;

        println!("IQ4_NL M={}: {:.2} µs/iter, {:.2} GFLOPS", m, avg_us, gflops);
    }
}
