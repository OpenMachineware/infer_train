use infer_train::backend::arm::gemv::*;
use infer_train::backend::arm::vec_dot::*;
use infer_train::types::*;
use std::time::Instant;

fn main() {
    println!("=== Q5_K GEMV Benchmark (matching model dimensions) ===\n");

    // Q projection: in_features=1024, out_features=2048
    let in_features = 1024;
    let out_features = 2048;
    let n_blocks = in_features / 256; // 4
    let block_size = std::mem::size_of::<BlockQ5_K>(); // 176

    println!("in_features: {}, out_features: {}", in_features, out_features);
    println!("n_blocks: {}, block_size: {} bytes", n_blocks, block_size);
    println!(
        "col_stride: {} bytes (should be {})",
        n_blocks * block_size,
        n_blocks * block_size
    );

    // Create test weights and input
    let weights: Vec<BlockQ5_K> = (0..out_features * n_blocks)
        .map(|i| BlockQ5_K {
            d: half::f16::from_f32(0.1 + (i % 10) as f32 * 0.01).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [((i % 16) + 1) as u8; 12],
            qh: [0; 32],
            qs: [((i % 256) + 1) as u8; 128],
        })
        .collect();

    let input: Vec<BlockQ8_K> = (0..n_blocks)
        .map(|i| BlockQ8_K {
            d: 1.0 + (i % 10) as f32 * 0.1,
            qs: [((i % 128) + 1) as i8; 256],
            bsums: [0; 16],
        })
        .collect();

    let mut output = vec![0.0f32; out_features];

    let weights_bytes = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * block_size,
        )
    };
    let col_stride = n_blocks * block_size;

    println!(
        "\nweights: {} blocks, {} bytes",
        weights.len(),
        weights_bytes.len()
    );
    println!("input: {} blocks", input.len());
    println!("col_stride: {} bytes\n", col_stride);

    // Warmup
    for _ in 0..5 {
        mul_mat_q5_k(
            weights_bytes,
            col_stride,
            &input,
            &mut output,
            out_features,
            in_features,
            1,
            1, // 1 thread
        );
    }

    // Benchmark single-thread
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        mul_mat_q5_k(
            weights_bytes,
            col_stride,
            &input,
            &mut output,
            out_features,
            in_features,
            1,
            1,
        );
    }
    let time = start.elapsed();
    let avg_us = time.as_micros() as f64 / iters as f64;

    let flops = 2 * out_features * in_features;
    let gflops = flops as f64 / avg_us / 1000.0;

    let weight_bytes = weights.len() * block_size;
    let input_bytes = input.len() * std::mem::size_of::<BlockQ8_K>();
    let output_bytes = output.len() * 4;
    let total_bytes = weight_bytes + input_bytes + output_bytes;
    let bandwidth_gbs = total_bytes as f64 / avg_us / 1000.0;

    println!("Q5_K GEMV (1 thread):");
    println!("  Time: {:.2} us", avg_us);
    println!("  Performance: {:.2} GFLOPS", gflops);
    println!("  Bandwidth: {:.2} GB/s", bandwidth_gbs);
    println!("  Memory: {:.2} MB", total_bytes as f64 / 1e6);

    // Benchmark multi-thread
    for n_threads in [2, 4, 8].iter() {
        let start = Instant::now();
        for _ in 0..iters {
            mul_mat_q5_k(
                weights_bytes,
                col_stride,
                &input,
                &mut output,
                out_features,
                in_features,
                1,
                *n_threads,
            );
        }
        let time = start.elapsed();
        let avg_us = time.as_micros() as f64 / iters as f64;
        let gflops = flops as f64 / avg_us / 1000.0;
        let bandwidth_gbs = total_bytes as f64 / avg_us / 1000.0;

        println!("\nQ5_K GEMV ({} threads):", n_threads);
        println!("  Time: {:.2} us", avg_us);
        println!("  Performance: {:.2} GFLOPS", gflops);
        println!("  Bandwidth: {:.2} GB/s", bandwidth_gbs);
    }
}
