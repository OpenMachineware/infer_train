use infer_train::quant::types::{BlockQ4K, QK_K};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;

fn main() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    // Test with M=8 to see all results clearly
    let m = 8;
    let k = 256;

    let nb = k / QK_K;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut scales = [0u8; 12];
        let mut qs = [0u8; 128];
        for j in 0..12 {
            scales[j] = ((i * 17 + j * 13) % 64) as u8;
        }
        for j in 0..128 {
            qs[j] = ((i * 23 + j * 17) % 256) as u8;
        }
        weights.push(BlockQ4K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            dmin: f16::from_f32(0.1 + (i % 5) as f32 * 0.05).to_bits(),
            scales,
            qs,
        });
    }

    let input: Vec<f32> = (0..k).map(|i| (i % 10) as f32 * 0.1).collect();

    let result = ctx.mv_q4_k_f32(m, k, &weights, &input).expect("GPU kernel failed");

    println!("M={}, K={} results:", m, k);
    for i in 0..m {
        println!("  row {}: {}", i, result[i]);
    }
}
