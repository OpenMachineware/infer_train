use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use infer_train::quant::types::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8_0};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;
use std::time::Instant;

fn generate_q2_k_weights(m: usize, k: usize) -> Vec<BlockQ2K> {
    let nb = k / 256;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut scales = [0u8; 16];
        let mut qs = [0u8; 64];
        for j in 0..16 {
            scales[j] = ((i + j + 1) | ((i + j) << 4)) as u8;
        }
        for j in 0..64 {
            qs[j] = ((i * 23 + j * 7) % 256) as u8;
        }
        weights.push(BlockQ2K {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qs,
        });
    }
    weights
}

fn generate_q3_k_weights(m: usize, k: usize) -> Vec<BlockQ3K> {
    let nb = k / 256;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut hmask = [0u8; 32];
        let mut qs = [0u8; 64];
        let mut scales = [0u8; 12];
        for j in 0..32 {
            hmask[j] = if j % 2 == 0 { 0xFF } else { 0x00 };
        }
        for j in 0..64 {
            qs[j] = ((i * 17 + j * 4) % 256) as u8;
        }
        for j in 0..12 {
            scales[j] = (40 + (j % 8)) as u8;
        }
        weights.push(BlockQ3K {
            d: f16::from_f32(1.0).to_bits(),
            hmask,
            qs,
            scales,
        });
    }
    weights
}

fn generate_q4_k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut qs = [0u8; 128];
        let mut scales = [0u8; 12];
        for j in 0..128 {
            qs[j] = ((i * 17 + j * 13) % 256) as u8;
        }
        for j in 0..12 {
            scales[j] = ((i + j) % 64) as u8;
        }
        weights.push(BlockQ4K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qs,
        });
    }
    weights
}

fn generate_q5_k_weights(m: usize, k: usize) -> Vec<BlockQ5K> {
    let nb = k / 256;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut qs = [0u8; 128];
        let mut qh = [0u8; 32];
        let mut scales = [0u8; 12];
        for j in 0..128 {
            qs[j] = ((i * 17 + j * 13) % 256) as u8;
        }
        for j in 0..32 {
            qh[j] = ((i * 23 + j * 7) % 256) as u8;
        }
        for j in 0..12 {
            scales[j] = ((i + j) % 64) as u8;
        }
        weights.push(BlockQ5K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qh,
            qs,
        });
    }
    weights
}

fn generate_q6_k_weights(m: usize, k: usize) -> Vec<BlockQ6K> {
    let nb = k / 256;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        let mut scales = [0i8; 16];
        for j in 0..128 {
            ql[j] = ((i * 17 + j * 13) % 256) as u8;
        }
        for j in 0..64 {
            qh[j] = ((i * 23 + j * 7) % 256) as u8;
        }
        for j in 0..16 {
            scales[j] = (40 + (j % 8) as i8) as i8;
        }
        weights.push(BlockQ6K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            ql,
            qh,
            scales,
        });
    }
    weights
}

fn generate_f32_input(k: usize) -> Vec<f32> {
    let mut input = Vec::with_capacity(k);
    for i in 0..k {
        input.push(1.0 + (i % 10) as f32 * 0.1);
    }
    input
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

fn calculate_gflops(m: usize, n: usize, k: usize, time_us: f64) -> f64 {
    let ops = 2.0 * m as f64 * n as f64 * k as f64;
    let time_s = time_us / 1_000_000.0;
    ops / time_s / 1_000_000_000.0
}

fn bench_gemm_all_formats(c: &mut Criterion) {
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let mut group = c.benchmark_group("gemm_comparison");

    let k_sizes: Vec<usize> = vec![256, 512, 1024, 2048, 4096, 8192, 16384];
    let m_sizes: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    let n_sizes: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

    // Test MV (M=1, various K sizes)
    println!("\n=== MV Performance (Decode Path) ===");
    for k in &k_sizes {
        let weights = generate_q4_k_weights(1, *k);
        let input = generate_f32_input(*k);

        let start = Instant::now();
        let result = ctx.mv_q4_k_f32(1, *k, &weights, &input).expect("GPU failed");
        let elapsed = start.elapsed().as_micros() as f64;

        let gflops = calculate_gflops(1, 1, *k, elapsed);
        println!("Q4_K MV K={}: {:.2} µs ({:.2} GFLOPS)", k, elapsed, gflops);

        group.bench_with_input(
            BenchmarkId::new("mv_q4_k", format!("K={}", k)),
            &k,
            |b, _| {
                b.iter(|| black_box(ctx.mv_q4_k_f32(1, *k, &weights, &input).unwrap()));
            },
        );
    }

    // Test GEMM for all K-quant formats
    println!("\n=== GEMM Performance (Batch Inference) ===");
    for k in &[4096] {
        for m in &[256, 512, 1024, 2048, 4096] {
            for n in &[1, 2, 4, 8, 16, 32, 64, 128] {
                let weights = generate_q4_k_weights(*m, *k);
                let input = generate_f32_input(*k * *n);

                let start = Instant::now();
                let result = ctx.gemm_q4_k_f32(*m, *n, *k, &weights, &input).expect("GPU failed");
                let elapsed = start.elapsed().as_micros() as f64;

                let gflops = calculate_gflops(*m, *n, *k, elapsed);
                println!("Q4_K GEMM M={} N={} K={}: {:.2} µs ({:.2} GFLOPS)", m, n, k, elapsed, gflops);

                group.bench_with_input(
                    BenchmarkId::new("gemm_q4_k", format!("{}x{}x{}", m, n, k)),
                    &(m, n, k),
                    |b, _| {
                        b.iter(|| black_box(ctx.gemm_q4_k_f32(*m, *n, *k, &weights, &input).unwrap()));
                    },
                );
            }
        }
    }

    // Test all K-quant formats for GEMM M=4096, N=128, K=4096
    println!("\n=== All K-Quant Formats GEMM (M=4096, N=128, K=4096) ===");
    let m = 4096;
    let n = 128;
    let k = 4096;

    let formats = vec![
        ("Q2_K", false),
        ("Q3_K", false),
        ("Q4_K", false),
        ("Q5_K", false),
        ("Q6_K", false),
    ];

    for (format, _is_mv) in formats {
        let elapsed = match format {
            "Q2_K" => {
                let weights = generate_q2_k_weights(m, k);
                let input = generate_f32_input(k * n);
                let start = Instant::now();
                let _ = ctx.gemm_q2_k_f32(m, n, k, &weights, &input).expect("GPU failed");
                start.elapsed().as_micros() as f64
            }
            "Q3_K" => {
                let weights = generate_q3_k_weights(m, k);
                let input = generate_f32_input(k * n);
                let start = Instant::now();
                let _ = ctx.gemm_q3_k_f32(m, n, k, &weights, &input).expect("GPU failed");
                start.elapsed().as_micros() as f64
            }
            "Q4_K" => {
                let weights = generate_q4_k_weights(m, k);
                let input = generate_f32_input(k * n);
                let start = Instant::now();
                let _ = ctx.gemm_q4_k_f32(m, n, k, &weights, &input).expect("GPU failed");
                start.elapsed().as_micros() as f64
            }
            "Q5_K" => {
                let weights = generate_q5_k_weights(m, k);
                let input = generate_f32_input(k * n);
                let start = Instant::now();
                let _ = ctx.gemm_q5_k_f32(m, n, k, &weights, &input).expect("GPU failed");
                start.elapsed().as_micros() as f64
            }
            "Q6_K" => {
                let weights = generate_q6_k_weights(m, k);
                let input = generate_f32_input(k * n);
                let start = Instant::now();
                let _ = ctx.gemm_q6_k_f32(m, n, k, &weights, &input).expect("GPU failed");
                start.elapsed().as_micros() as f64
            }
            _ => 0.0,
        };

        let gflops = calculate_gflops(m, n, k, elapsed);
        println!("{} GEMM: {:.2} µs ({:.2} GFLOPS)", format, elapsed, gflops);
    }

    // Test small K optimization
    println!("\n=== Small K Performance ===");
    for k in &[256, 512, 1024, 2048] {
        let m = 16;
        let n = 16;

        let weights = generate_q4_k_weights(m, *k);
        let input = generate_f32_input(*k * n);

        let start = Instant::now();
        let _ = ctx.gemm_q4_k_f32(m, n, *k, &weights, &input).expect("GPU failed");
        let elapsed = start.elapsed().as_micros() as f64;

        let gflops = calculate_gflops(m, n, *k, elapsed);
        println!("Q4_K GEMM M={} N={} K={}: {:.2} µs ({:.2} GFLOPS)", m, n, k, elapsed, gflops);
    }

    group.finish();
}

fn bench_mv_all_formats(c: &mut Criterion) {
    let ctx = MetalContext::new().expect("Failed to create Metal context");
    let mut group = c.benchmark_group("mv_all_formats");

    let m_sizes: Vec<usize> = vec![256, 512, 1024, 2048, 4096, 8192];
    let k = 4096;

    println!("\n=== MV Performance for All K-Quant Formats ===");

    for m in &m_sizes {
        println!("\n--- M={} ---", m);

        // Q4_K
        {
            let weights = generate_q4_k_weights(*m, k);
            let input = generate_f32_input(k);
            let start = Instant::now();
            let _ = ctx.mv_q4_k_f32(*m, k, &weights, &input).expect("GPU failed");
            let elapsed = start.elapsed().as_micros() as f64;
            let gflops = calculate_gflops(*m, 1, k, elapsed);
            println!("Q4_K: {:.2} µs ({:.2} GFLOPS)", elapsed, gflops);
        }

        // Q2_K
        {
            let weights = generate_q2_k_weights(*m, k);
            let input = generate_f32_input(k);
            let start = Instant::now();
            let _ = ctx.mv_q2_k_f32(*m, k, &weights, &input).expect("GPU failed");
            let elapsed = start.elapsed().as_micros() as f64;
            let gflops = calculate_gflops(*m, 1, k, elapsed);
            println!("Q2_K: {:.2} µs ({:.2} GFLOPS)", elapsed, gflops);
        }

        // Q3_K
        {
            let weights = generate_q3_k_weights(*m, k);
            let input = generate_f32_input(k);
            let start = Instant::now();
            let _ = ctx.mv_q3_k_f32(*m, k, &weights, &input).expect("GPU failed");
            let elapsed = start.elapsed().as_micros() as f64;
            let gflops = calculate_gflops(*m, 1, k, elapsed);
            println!("Q3_K: {:.2} µs ({:.2} GFLOPS)", elapsed, gflops);
        }

        // Q5_K
        {
            let weights = generate_q5_k_weights(*m, k);
            let input = generate_f32_input(k);
            let start = Instant::now();
            let _ = ctx.mv_q5_k_f32(*m, k, &weights, &input).expect("GPU failed");
            let elapsed = start.elapsed().as_micros() as f64;
            let gflops = calculate_gflops(*m, 1, k, elapsed);
            println!("Q5_K: {:.2} µs ({:.2} GFLOPS)", elapsed, gflops);
        }

        // Q6_K
        {
            let weights = generate_q6_k_weights(*m, k);
            let input = generate_f32_input(k);
            let start = Instant::now();
            let _ = ctx.mv_q6_k_f32(*m, k, &weights, &input).expect("GPU failed");
            let elapsed = start.elapsed().as_micros() as f64;
            let gflops = calculate_gflops(*m, 1, k, elapsed);
            println!("Q6_K: {:.2} µs ({:.2} GFLOPS)", elapsed, gflops);
        }
    }

    group.finish();
}

criterion_group!(benches, bench_gemm_all_formats, bench_mv_all_formats);
criterion_main!(benches);
