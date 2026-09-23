use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use infer_train::quant::types::{BlockIQ4NL, BlockQ8_0};
use infer_train::quant::vec_dot::arm::vec_dot_iq4_nl_q8_0_neon;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;

fn generate_iq4_nl_weights(m: usize, k: usize) -> Vec<BlockIQ4NL> {
    let nb = k / 32;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut qs = [0u8; 16];
        for j in 0..16 {
            qs[j] = ((i * 17 + j * 13) % 256) as u8;
        }
        weights.push(BlockIQ4NL {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
        });
    }
    weights
}

fn generate_q8_0_input(k: usize) -> Vec<BlockQ8_0> {
    let nb = k / 32;
    let mut input = Vec::with_capacity(nb);
    for i in 0..nb {
        let mut qs = [0i8; 32];
        for j in 0..32 {
            let val = ((i * 23 + j * 7) % 256) as i32 - 128;
            qs[j] = val as i8;
        }
        input.push(BlockQ8_0 {
            d: f16::from_f32(1.0 + (i % 10) as f32 * 0.1).to_bits(),
            qs,
        });
    }
    input
}

fn mv_cpu(m: usize, k: usize, weights: &[BlockIQ4NL], input: &[BlockQ8_0]) -> Vec<f32> {
    let nb = k / 32;
    let mut output = Vec::with_capacity(m);

    for row in 0..m {
        let row_weights = &weights[row * nb..(row + 1) * nb];
        let result = unsafe { vec_dot_iq4_nl_q8_0_neon(k, row_weights, input) };
        output.push(result);
    }

    output
}

fn bench_mv_gpu(c: &mut Criterion) {
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let mut group = c.benchmark_group("mv_iq4_nl");

    for &(m, k) in &[(256, 4096), (1024, 4096), (2048, 4096)] {
        let weights = generate_iq4_nl_weights(m, k);
        let input = generate_q8_0_input(k);

        // Verify correctness first
        let cpu_result = mv_cpu(m, k, &weights, &input);
        let gpu_result = ctx.mv_iq4_nl_q8_0(m, k, &weights, &input).expect("GPU failed");

        let max_diff = cpu_result.iter()
            .zip(gpu_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("MV {}x{} correctness: max_diff = {}", m, k, max_diff);

        group.bench_with_input(BenchmarkId::new("cpu_neon", format!("{}x{}", m, k)), &m, |b, _| {
            b.iter(|| black_box(mv_cpu(m, k, &weights, &input)));
        });

        group.bench_with_input(BenchmarkId::new("gpu_metal", format!("{}x{}", m, k)), &m, |b, _| {
            b.iter(|| black_box(ctx.mv_iq4_nl_q8_0(m, k, &weights, &input).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("gpu_simd", format!("{}x{}", m, k)), &m, |b, _| {
            b.iter(|| black_box(ctx.mv_iq4_nl_q8_0_simd(m, k, &weights, &input).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_mv_gpu);
criterion_main!(benches);
