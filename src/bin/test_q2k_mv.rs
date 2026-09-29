// Q2_K MV test
use infer_train::quant::types::BlockQ2K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use std::time::Instant;

fn main() {
    let ctx = MetalContext::new().expect("Failed to create MetalContext");
    let m = 4096usize;
    let k = 4096usize;
    let nb = m * (k / 256);

    let weights: Vec<BlockQ2K> = (0..nb)
        .map(|i| {
            let mut scales = [0u8; 16];
            let mut qs = [0u8; 64];
            for j in 0..16 {
                scales[j] = ((j + 1) % 8) as u8;
            }
            for j in 0..64 {
                qs[j] = ((i + j + 42) & 0xFF) as u8;
            }
            BlockQ2K {
                d: half::f16::from_f32(1.0).to_bits(),
                dmin: half::f16::from_f32(0.0).to_bits(),
                scales,
                qs,
            }
        })
        .collect();

    let input = vec![1.0f32; k];

    let mut times = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        let _ = ctx.mv_q2_k_f32(m, k, &weights, &input).unwrap();
        times.push(start.elapsed().as_nanos() as f64);
    }

    let avg_ns = times.iter().sum::<f64>() / 10.0;
    let gflops = 2.0 * (m as f64) * (k as f64) / (avg_ns / 1e9) / 1e9;
    println!("Q2_K MV: {:.2} GFLOPS", gflops);
}
