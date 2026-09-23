// Comprehensive multi-size benchmark for all IQ/TQ Metal MV kernels
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
        for j in 0..16 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        BlockIQ4NL { d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(), qs }
    }).collect()
}

fn generate_iq1_s_weights(m: usize, k: usize) -> Vec<BlockIQ1S> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 32];
        let mut qh = [0u16; 8];
        for j in 0..32 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..8 { qh[j] = ((i * 19 + j * 11) % 65536) as u16; }
        BlockIQ1S { d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(), qs, qh }
    }).collect()
}

fn generate_iq1_m_weights(m: usize, k: usize) -> Vec<BlockIQ1M> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 32];
        let mut qh = [0u8; 16];
        let mut scales = [0u8; 8];
        for j in 0..32 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..16 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        for j in 0..8 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockIQ1M { qs, qh, scales }
    }).collect()
}

fn generate_iq2_xxs_weights(m: usize, k: usize) -> Vec<BlockIQ2XXS> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u16; 32];
        for j in 0..32 { qs[j] = ((i * 17 + j * 13) % 65536) as u16; }
        BlockIQ2XXS { d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(), qs }
    }).collect()
}

fn generate_iq2_xs_weights(m: usize, k: usize) -> Vec<BlockIQ2XS> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u16; 32];
        let mut scales = [0u8; 8];
        for j in 0..32 { qs[j] = ((i * 17 + j * 13) % 65536) as u16; }
        for j in 0..8 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockIQ2XS { d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(), qs, scales }
    }).collect()
}

fn generate_iq2_s_weights(m: usize, k: usize) -> Vec<BlockIQ2S> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 64];
        let mut qh = [0u8; 8];
        let mut scales = [0u8; 8];
        for j in 0..64 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..8 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        for j in 0..8 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockIQ2S { d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(), qs, qh, scales }
    }).collect()
}

fn generate_iq3_xxs_weights(m: usize, k: usize) -> Vec<BlockIQ3XXS> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 96];
        for j in 0..96 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        BlockIQ3XXS { d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(), qs }
    }).collect()
}

fn generate_iq3_s_weights(m: usize, k: usize) -> Vec<BlockIQ3S> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 64];
        let mut qh = [0u8; 8];
        let mut signs = [0u8; 32];
        let mut scales = [0u8; 4];
        for j in 0..64 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..8 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        for j in 0..32 { signs[j] = ((i * 23 + j * 7) % 256) as u8; }
        for j in 0..4 { scales[j] = ((i * 29 + j * 5) % 256) as u8; }
        BlockIQ3S { d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(), qs, qh, signs, scales }
    }).collect()
}

fn generate_tq2_0_weights(m: usize, k: usize) -> Vec<BlockTQ2_0> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 64];
        for j in 0..64 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        BlockTQ2_0 { d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(), qs }
    }).collect()
}

fn generate_tq1_0_weights(m: usize, k: usize) -> Vec<BlockTQ1_0> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 48];
        let mut qh = [0u8; 4];
        for j in 0..48 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..4 { qh[j] = ((i * 19 + j * 11) % 256) as u8; }
        BlockTQ1_0 { qs, qh, d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits() }
    }).collect()
}

fn bench_format<F, W>(
    ctx: &MetalContext, name: &str, m: usize, k: usize,
    weights: &[W], input: &[f32], method: F, n_iter: usize
) -> f64 where F: Fn(&MetalContext, usize, usize, &[W], &[f32]) -> Result<Vec<f32>, String> {
    // Warmup
    let _ = method(ctx, m, k, weights, input).expect("GPU failed");

    let start = std::time::Instant::now();
    for _ in 0..n_iter {
        let _ = method(ctx, m, k, weights, input).expect("GPU failed");
    }
    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / n_iter as f64;
    let gflops = (2.0 * m as f64 * k as f64) / (avg_us * 1e-6) / 1e9;

    println!("{:10} K={:5}: {:8.2} µs, {:6.2} GFLOPS", name, k, avg_us, gflops);
    gflops
}

fn main() {
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let m = 1024;
    let sizes = [256, 512, 1024, 2048, 4096];

    println!("=== IQ/TQ Metal MV Multi-Size Benchmark ===");
    println!("M={}\n", m);

    for k in &sizes {
        let input = generate_f32_input(*k);

        // IQ4_NL
        let weights = generate_iq4_nl_weights(m, *k);
        bench_format(&ctx, "IQ4_NL", m, *k, &weights, &input, MetalContext::mv_iq4_nl_f32, 500);

        // IQ1_S
        let weights = generate_iq1_s_weights(m, *k);
        bench_format(&ctx, "IQ1_S", m, *k, &weights, &input, MetalContext::mv_iq1_s_f32, 500);

        // IQ1_M
        let weights = generate_iq1_m_weights(m, *k);
        bench_format(&ctx, "IQ1_M", m, *k, &weights, &input, MetalContext::mv_iq1_m_f32, 500);

        // IQ2_XXS
        let weights = generate_iq2_xxs_weights(m, *k);
        bench_format(&ctx, "IQ2_XXS", m, *k, &weights, &input, MetalContext::mv_iq2_xxs_f32, 500);

        // IQ2_XS
        let weights = generate_iq2_xs_weights(m, *k);
        bench_format(&ctx, "IQ2_XS", m, *k, &weights, &input, MetalContext::mv_iq2_xs_f32, 500);

        // IQ2_S
        let weights = generate_iq2_s_weights(m, *k);
        bench_format(&ctx, "IQ2_S", m, *k, &weights, &input, MetalContext::mv_iq2_s_f32, 500);

        // IQ3_XXS
        let weights = generate_iq3_xxs_weights(m, *k);
        bench_format(&ctx, "IQ3_XXS", m, *k, &weights, &input, MetalContext::mv_iq3_xxs_f32, 500);

        // IQ3_S
        let weights = generate_iq3_s_weights(m, *k);
        bench_format(&ctx, "IQ3_S", m, *k, &weights, &input, MetalContext::mv_iq3_s_f32, 500);

        // TQ2_0
        let weights = generate_tq2_0_weights(m, *k);
        bench_format(&ctx, "TQ2_0", m, *k, &weights, &input, MetalContext::mv_tq2_0_f32, 500);

        // TQ1_0
        let weights = generate_tq1_0_weights(m, *k);
        bench_format(&ctx, "TQ1_0", m, *k, &weights, &input, MetalContext::mv_tq1_0_f32, 500);

        println!();
    }
}
