// Benchmark Rayon threading overhead vs sequential execution

use rayon::prelude::*;

fn compute_sequential(data: &[i32]) -> i32 {
    data.iter().sum()
}

fn compute_parallel(data: &[i32]) -> i32 {
    data.par_iter().sum()
}

fn main() {
    println!("=== Rayon Threading Overhead Benchmark ===\n");

    let sizes = [100, 1000, 10_000, 100_000, 1_000_000];

    for &size in &sizes {
        let data: Vec<i32> = (0..size as i32).collect();

        // Warm up Rayon
        let _ = data.par_iter().sum::<i32>();

        // Sequential
        let n_iter = 10_000;
        let start = std::time::Instant::now();
        for _ in 0..n_iter {
            let _ = compute_sequential(&data);
        }
        let seq_time_us = start.elapsed().as_micros() as f64 / n_iter as f64;

        // Parallel
        let start = std::time::Instant::now();
        for _ in 0..n_iter {
            let _ = compute_parallel(&data);
        }
        let par_time_us = start.elapsed().as_micros() as f64 / n_iter as f64;

        let speedup = seq_time_us / par_time_us;
        println!(
            "Size {}: seq={:.2} µs, par={:.2} µs, speedup={:.2}x",
            size, seq_time_us, par_time_us, speedup
        );
    }

    println!("\n=== Small GEMV Simulation ===\n");

    // Simulate Q4_K GEMV (1024 x 2048)
    // Each row is 1024 elements, we have 2048 rows
    // Each vec_dot computes 1024 dot products

    let k = 1024;
    let m = 2048;

    // Create test data (simulating quantized weights and input)
    let weights: Vec<i32> =
        (0..m * k).map(|i| (i % 256 - 128) as i32).collect();
    let input: Vec<i32> = (0..k).map(|i| (i % 128 - 64) as i32).collect();
    let mut output = vec![0i32; m];

    // Sequential GEMV
    let n_iter = 100;
    let start = std::time::Instant::now();
    for _ in 0..n_iter {
        for i in 0..m {
            let mut sum = 0i32;
            for j in 0..k {
                sum += weights[i * k + j] * input[j];
            }
            output[i] = sum;
        }
    }
    let seq_time_us = start.elapsed().as_micros() as f64 / n_iter as f64;

    // Parallel GEMV with Rayon
    let start = std::time::Instant::now();
    for _ in 0..n_iter {
        output.par_iter_mut().enumerate().for_each(|(i, out)| {
            let mut sum = 0i32;
            for j in 0..k {
                sum += weights[i * k + j] * input[j];
            }
            *out = sum;
        });
    }
    let par_time_us = start.elapsed().as_micros() as f64 / n_iter as f64;

    let speedup = seq_time_us / par_time_us;
    println!(
        "GEMV ({}x{}): seq={:.2} µs, par={:.2} µs, speedup={:.2}x",
        k, m, seq_time_us, par_time_us, speedup
    );

    // Calculate throughput
    let flops = 2 * m * k; // multiply + add for each element
    let gflops_seq = flops as f64 / seq_time_us / 1000.0;
    let gflops_par = flops as f64 / par_time_us / 1000.0;
    println!("Sequential: {:.2} GFLOPS", gflops_seq);
    println!("Parallel: {:.2} GFLOPS", gflops_par);
}
