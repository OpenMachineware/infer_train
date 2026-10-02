// Compare all K-quant MV kernels using MetalContext
use half::f16;
use infer_train::quant::types::*;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use std::time::Instant;

fn generate_q4k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
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
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qs,
            }
        })
        .collect()
}

fn generate_q2k_weights(m: usize, k: usize) -> Vec<BlockQ2K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut scales = [0u8; 16];
            let mut qs = [0u8; 64];
            for j in 0..16 {
                scales[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            for j in 0..64 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            BlockQ2K {
                scales,
                qs,
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
            }
        })
        .collect()
}

fn generate_q3k_weights(m: usize, k: usize) -> Vec<BlockQ3K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut hmask = [0u8; 32];
            let mut qs = [0u8; 64];
            let mut scales = [0u8; 12];
            for j in 0..32 {
                hmask[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            for j in 0..64 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 23 + j * 7) % 256) as u8;
            }
            BlockQ3K {
                hmask,
                qs,
                scales,
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            }
        })
        .collect()
}

fn generate_q5k_weights(m: usize, k: usize) -> Vec<BlockQ5K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut qs = [0u8; 128];
            let mut qh = [0u8; 32];
            let mut scales = [0u8; 12];
            for j in 0..128 {
                qs[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..32 {
                qh[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            for j in 0..12 {
                scales[j] = ((i * 23 + j * 7) % 256) as u8;
            }
            BlockQ5K {
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
                dmin: f16::from_f32(0.1).to_bits(),
                scales,
                qh,
                qs,
            }
        })
        .collect()
}

fn generate_q6k_weights(m: usize, k: usize) -> Vec<BlockQ6K> {
    let nb = k / 256;
    (0..m * nb)
        .map(|i| {
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0i8; 16];
            for j in 0..128 {
                ql[j] = ((i * 17 + j * 13) % 256) as u8;
            }
            for j in 0..64 {
                qh[j] = ((i * 19 + j * 11) % 256) as u8;
            }
            for j in 0..16 {
                scales[j] = ((i * 23 + j * 7) % 128) as i8 - 64;
            }
            BlockQ6K {
                ql,
                qh,
                scales,
                d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            }
        })
        .collect()
}

fn generate_input(k: usize) -> Vec<f32> {
    (0..k).map(|i| (i % 10) as f32 * 0.1).collect()
}

fn main() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    let m = 4096usize;
    let k = 4096usize;
    let iterations = 5000;
    let warmup = 100;

    let input = generate_input(k);

    println!(
        "=== K-quant MV Performance (M={}, K={}, {} iter) ===\n",
        m, k, iterations
    );
    println!("{:8} | {:>10} | {:>10}", "Format", "Time (us)", "GFLOPS");

    // Q4_K
    {
        let weights = generate_q4k_weights(m, k);
        for _ in 0..warmup {
            let _ = ctx.mv_q4_k_f32(m, k, &weights, &input);
        }
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ctx.mv_q4_k_f32(m, k, &weights, &input);
        }
        let elapsed = start.elapsed().as_micros() as f64 / iterations as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (elapsed * 1e-6) / 1e9;
        println!("{:8} | {:>10.2} | {:>10.1}", "Q4_K", elapsed, gflops);
    }

    // Q2_K
    {
        let weights = generate_q2k_weights(m, k);
        for _ in 0..warmup {
            let _ = ctx.mv_q2_k_f32(m, k, &weights, &input);
        }
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ctx.mv_q2_k_f32(m, k, &weights, &input);
        }
        let elapsed = start.elapsed().as_micros() as f64 / iterations as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (elapsed * 1e-6) / 1e9;
        println!("{:8} | {:>10.2} | {:>10.1}", "Q2_K", elapsed, gflops);
    }

    // Q3_K
    {
        let weights = generate_q3k_weights(m, k);
        for _ in 0..warmup {
            let _ = ctx.mv_q3_k_f32(m, k, &weights, &input);
        }
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ctx.mv_q3_k_f32(m, k, &weights, &input);
        }
        let elapsed = start.elapsed().as_micros() as f64 / iterations as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (elapsed * 1e-6) / 1e9;
        println!("{:8} | {:>10.2} | {:>10.1}", "Q3_K", elapsed, gflops);
    }

    // Q5_K
    {
        let weights = generate_q5k_weights(m, k);
        for _ in 0..warmup {
            let _ = ctx.mv_q5_k_f32(m, k, &weights, &input);
        }
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ctx.mv_q5_k_f32(m, k, &weights, &input);
        }
        let elapsed = start.elapsed().as_micros() as f64 / iterations as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (elapsed * 1e-6) / 1e9;
        println!("{:8} | {:>10.2} | {:>10.1}", "Q5_K", elapsed, gflops);
    }

    // Q6_K
    {
        let weights = generate_q6k_weights(m, k);
        for _ in 0..warmup {
            let _ = ctx.mv_q6_k_f32(m, k, &weights, &input);
        }
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = ctx.mv_q6_k_f32(m, k, &weights, &input);
        }
        let elapsed = start.elapsed().as_micros() as f64 / iterations as f64;
        let gflops = (2.0 * m as f64 * k as f64) / (elapsed * 1e-6) / 1e9;
        println!("{:8} | {:>10.2} | {:>10.1}", "Q6_K", elapsed, gflops);
    }
}
