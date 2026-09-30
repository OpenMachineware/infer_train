// Benchmark argsort at different sizes
use infer_train::ops::sampling::argsort_f32_inplace;
use std::time::Instant;

fn bench_size(n: usize, iterations: usize) -> f64 {
    let data: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let mut indices = vec![0i32; n];

    let start = Instant::now();
    for _ in 0..iterations {
        argsort_f32_inplace(&data, &mut indices);
        std::hint::black_box(&indices);
    }
    start.elapsed().as_secs_f64() * 1e6 / iterations as f64
}

fn main() {
    // MoE sizes (8, 16, 32, 64) + other sizes
    let sizes = [
        (8, 100000),
        (16, 50000),
        (32, 20000),
        (64, 10000),
        (100, 10000),
        (500, 10000),
        (1000, 10000),
        (2000, 5000),
        (5000, 1000),
        (10000, 1000),
    ];

    for (n, iter) in sizes {
        let time = bench_size(n, iter);
        println!("Rust n={}: {:.2} us", n, time);
    }
}
