use infer_train::quant::types::{BlockQ2K, BlockQ6K, BlockQ3K};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

#[test]
fn bench_metal_q2_k_scaling() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    let k = 4096;
    let nb = k / 256;

    for &m in &[64, 128, 256, 512, 1024, 2048, 4096, 8192] {
        let mut x_blocks = Vec::with_capacity(m * nb);
        let mut input = vec![0.0f32; k];

        for i in 0..m * nb {
            let mut qs = [0u8; 64];
            for j in 0..64 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            let mut scales = [0u8; 16];
            for j in 0..16 {
                scales[j] = ((i * 23 + j * 11) % 64) as u8;
            }
            x_blocks.push(BlockQ2K {
                d: half::f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: half::f16::from_f32(0.1 + (i % 5) as f32 * 0.05).to_bits(),
                scales,
                qs,
            });
        }

        for i in 0..k {
            input[i] = (i % 10) as f32 * 0.1;
        }

        let _ = ctx.mv_q2_k_f32(m, k, &x_blocks, &input).expect("GPU failed");

        let n_iter = 100;
        let start = std::time::Instant::now();

        for _ in 0..n_iter {
            let _ = ctx.mv_q2_k_f32(m, k, &x_blocks, &input).expect("GPU failed");
        }

        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / n_iter as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (avg_us * 1e-6) / 1e9;

        println!("Q2_K M={}: {:.2} µs/iter, {:.2} GFLOPS", m, avg_us, gflops);
    }
}

#[test]
fn bench_metal_q6_k_scaling() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    let k = 4096;
    let nb = k / 256;

    for &m in &[64, 128, 256, 512, 1024, 2048, 4096, 8192] {
        let mut x_blocks = Vec::with_capacity(m * nb);
        let mut input = vec![0.0f32; k];

        for i in 0..m * nb {
            let mut ql = [0u8; 128];
            for j in 0..128 {
                ql[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            let mut qh = [0u8; 64];
            for j in 0..64 {
                qh[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            let mut scales = [0i8; 16];
            for j in 0..16 {
                scales[j] = ((i * 23 + j * 11) % 64) as i8;
            }
            x_blocks.push(BlockQ6K {
                ql,
                qh,
                scales,
                d: half::f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            });
        }

        for i in 0..k {
            input[i] = (i % 10) as f32 * 0.1;
        }

        let _ = ctx.mv_q6_k_f32(m, k, &x_blocks, &input).expect("GPU failed");

        let n_iter = 100;
        let start = std::time::Instant::now();

        for _ in 0..n_iter {
            let _ = ctx.mv_q6_k_f32(m, k, &x_blocks, &input).expect("GPU failed");
        }

        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / n_iter as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (avg_us * 1e-6) / 1e9;

        println!("Q6_K M={}: {:.2} µs/iter, {:.2} GFLOPS", m, avg_us, gflops);
    }
}

#[test]
fn bench_metal_q3_k_scaling() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    let k = 4096;
    let nb = k / 256;

    for &m in &[64, 128, 256, 512, 1024, 2048, 4096, 8192] {
        let mut x_blocks = Vec::with_capacity(m * nb);
        let mut input = vec![0.0f32; k];

        for i in 0..m * nb {
            let mut hmask = [0u8; 32];
            for j in 0..32 {
                hmask[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            let mut qs = [0u8; 64];
            for j in 0..64 {
                qs[j] = ((i * 23 + j * 17) % 256) as u8;
            }
            let mut scales = [0u8; 12];
            for j in 0..12 {
                scales[j] = ((i * 23 + j * 11) % 64) as u8;
            }
            x_blocks.push(BlockQ3K {
                d: half::f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                hmask,
                qs,
                scales,
            });
        }

        for i in 0..k {
            input[i] = (i % 10) as f32 * 0.1;
        }

        let _ = ctx.mv_q3_k_f32(m, k, &x_blocks, &input).expect("GPU failed");

        let n_iter = 100;
        let start = std::time::Instant::now();

        for _ in 0..n_iter {
            let _ = ctx.mv_q3_k_f32(m, k, &x_blocks, &input).expect("GPU failed");
        }

        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / n_iter as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (avg_us * 1e-6) / 1e9;

        println!("Q3_K M={}: {:.2} µs/iter, {:.2} GFLOPS", m, avg_us, gflops);
    }
}
