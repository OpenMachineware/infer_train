use infer_train::backend::arm::gemv::*;
use infer_train::types::*;
use std::time::Instant;

fn main() {
    let in_features = 1024;
    let out_features = 2048;
    let n_blocks = in_features / 256;
    let block_size = std::mem::size_of::<BlockQ5_K>();

    let weights: Vec<BlockQ5_K> = (0..out_features * n_blocks)
        .map(|i| BlockQ5_K {
            d: half::f16::from_f32(0.1).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [1u8; 12],
            qh: [0; 32],
            qs: [0x88; 128],
        })
        .collect();

    let input: Vec<BlockQ8_K> =
        vec![BlockQ8_K { d: 1.0, qs: [1; 256], bsums: [0; 16] }; n_blocks];
    let mut output = vec![0.0f32; out_features];

    let weights_bytes = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * block_size,
        )
    };
    let col_stride = n_blocks * block_size;

    // First, warm up thread pool with multi-threaded calls
    println!("=== Warming up thread pool ===");
    for i in 0..100 {
        mul_mat_q5_k(
            weights_bytes,
            col_stride,
            &input,
            &mut output,
            out_features,
            in_features,
            1,
            8, // 8 threads
        );
        if i % 20 == 0 {
            println!("Warmup iteration {}", i);
        }
    }
    println!("Warmup done\n");

    // Now benchmark multi-thread
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
            8,
        );
    }
    let time = start.elapsed();
    let avg_us = time.as_micros() as f64 / iters as f64;

    let flops = 2 * out_features * in_features;
    let gflops = flops as f64 / avg_us / 1000.0;
    let total_bytes = weights.len() * block_size
        + input.len() * std::mem::size_of::<BlockQ8_K>()
        + output.len() * 4;
    let bandwidth_gbs = total_bytes as f64 / avg_us / 1000.0;

    println!("After warmup (8 threads):");
    println!("  Time: {:.2} us", avg_us);
    println!("  Performance: {:.2} GFLOPS", gflops);
    println!("  Bandwidth: {:.2} GB/s", bandwidth_gbs);

    // Compare with single-thread
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
            1, // 1 thread
        );
    }
    let time = start.elapsed();
    let avg_us = time.as_micros() as f64 / iters as f64;
    let gflops = flops as f64 / avg_us / 1000.0;
    let bandwidth_gbs = total_bytes as f64 / avg_us / 1000.0;

    println!("\nSingle-thread:");
    println!("  Time: {:.2} us", avg_us);
    println!("  Performance: {:.2} GFLOPS", gflops);
    println!("  Bandwidth: {:.2} GB/s", bandwidth_gbs);
}
