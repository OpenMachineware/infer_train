use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use infer_train::quant::types::{BlockIQ4NL, BlockQ8_0, QK4_0 as QK};
use infer_train::quant::vec_dot::arm::vec_dot_iq4_nl_q8_0_neon;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;

fn generate_iq4_nl_data(nb: usize) -> Vec<BlockIQ4NL> {
    let mut blocks = Vec::with_capacity(nb);
    for i in 0..nb {
        let mut qs = [0u8; 16];
        for j in 0..16 {
            qs[j] = ((i + j) % 256) as u8;
        }
        blocks.push(BlockIQ4NL {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
        });
    }
    blocks
}

fn generate_q8_0_data(nb: usize) -> Vec<BlockQ8_0> {
    let mut blocks = Vec::with_capacity(nb);
    for i in 0..nb {
        let mut qs = [0i8; 32];
        for j in 0..32 {
            qs[j] = ((i + j) % 256 - 128) as i8;
        }
        blocks.push(BlockQ8_0 {
            d: f16::from_f32(1.0 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
        });
    }
    blocks
}

fn bench_gpu_vs_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec_dot_iq4_nl_gpu");

    // Create GPU context
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    // Test with different sizes
    for &nb in &[64, 256, 1024] {
        let n = nb * QK;
        let x = generate_iq4_nl_data(nb);
        let y = generate_q8_0_data(nb);

        group.bench_with_input(BenchmarkId::new("cpu_neon", nb), &nb, |b, _| {
            b.iter(|| {
                unsafe {
                    black_box(vec_dot_iq4_nl_q8_0_neon(n, &x, &y))
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("gpu_metal", nb), &nb, |b, _| {
            b.iter(|| {
                black_box(ctx.vec_dot_iq4_nl_q8_0(n, &x, &y).unwrap())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_gpu_vs_cpu);
criterion_main!(benches);
