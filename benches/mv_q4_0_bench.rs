use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use infer_train::quant::types::BlockQ4_0;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;

// Data generation - MUST match test_llama_metal_mv.cpp exactly
fn generate_q4_0_weights(m: usize, k: usize) -> Vec<BlockQ4_0> {
    let nb = k / 32;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut qs = [0u8; 16];
        for j in 0..16 {
            qs[j] = ((i * 17 + j * 13) % 256) as u8;
        }
        weights.push(BlockQ4_0 {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
        });
    }
    weights
}

fn generate_f32_input(k: usize) -> Vec<f32> {
    let mut input = Vec::with_capacity(k);
    for i in 0..k {
        input.push((i % 10) as f32 * 0.1);
    }
    input
}

fn bench_mv_q4_0_gpu(c: &mut Criterion) {
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let mut group = c.benchmark_group("mv_q4_0_f32");

    for &(m, k) in &[(256, 4096), (1024, 4096), (2048, 4096), (8192, 4096)] {
        let weights = generate_q4_0_weights(m, k);
        let input = generate_f32_input(k);

        group.bench_with_input(BenchmarkId::new("rust_metal", format!("{}x{}", m, k)), &m, |b, _| {
            b.iter(|| black_box(ctx.mv_q4_0_f32(m, k, &weights, &input).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_mv_q4_0_gpu);
criterion_main!(benches);
