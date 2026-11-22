use infer_train::backend::arm::gemv::{
    gemv_q5_k_transposed, gemv_q5_k_x8_neon,
};
use infer_train::backend::arm::quant::quantize_q8_k;
use infer_train::backend::arm::repack::repack_q5_k_to_x8;
use infer_train::types::{BlockQ5_K, BlockQ5_Kx8, BlockQ8_K, QK_K};
use std::time::Instant;

fn main() {
    println!("=== Q5_K GEMV Comparison Benchmark ===\n");

    // Dimensions matching a typical layer
    let in_features = 4096;
    let out_features = 4096;
    let n_blocks = in_features / QK_K;

    // Create random-ish weights
    let mut weights: Vec<BlockQ5_K> =
        Vec::with_capacity(out_features * n_blocks);
    for row in 0..out_features {
        for blk in 0..n_blocks {
            let scale_val = 1.0 + (row as f32 % 10.0) * 0.1;
            let min_val = 0.1 + (blk as f32 % 5.0) * 0.05;

            let mut scales = [0u8; 12];
            for i in 0..4 {
                scales[i] = 32 + (row as u8 % 16) as u8;
            }
            for i in 4..8 {
                scales[i] = 16 + (blk as u8 % 8) as u8;
            }

            let mut qh = [0u8; 32];
            let mut qs = [0u8; 128];
            for i in 0..32 {
                qh[i] = ((row + i) % 256) as u8;
            }
            for i in 0..128 {
                qs[i] = ((row + blk + i) % 256) as u8;
            }

            weights.push(BlockQ5_K {
                d: half::f16::from_f32(scale_val).to_bits(),
                dmin: half::f16::from_f32(min_val).to_bits(),
                scales,
                qh,
                qs,
            });
        }
    }

    // Create input
    let mut input_f32: Vec<f32> =
        (0..in_features).map(|i| (i as f32 % 100.0 - 50.0) / 100.0).collect();
    let mut input_q8k: Vec<BlockQ8_K> =
        vec![unsafe { std::mem::zeroed() }; n_blocks];
    quantize_q8_k(&input_f32, &mut input_q8k);

    // Create output buffers
    let mut output_std = vec![0.0f32; out_features];
    let mut output_x8 = vec![0.0f32; out_features];

    // Calculate stride
    let stride = n_blocks * std::mem::size_of::<BlockQ5_K>();
    let weights_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * std::mem::size_of::<BlockQ5_K>(),
        )
    };

    // Repack weights for interleaved GEMV
    println!("Repacking weights...");
    let start_repack = Instant::now();
    let n_row_groups = out_features / 8;
    let mut weights_packed: Vec<u8> =
        vec![0u8; n_row_groups * n_blocks * std::mem::size_of::<BlockQ5_Kx8>()];
    repack_q5_k_to_x8(
        weights_bytes,
        stride,
        &mut weights_packed,
        out_features,
        n_blocks,
    );
    let repack_time = start_repack.elapsed();
    println!("Repack time: {:.2?}", repack_time);

    let weights_x8: &[BlockQ5_Kx8] = unsafe {
        std::slice::from_raw_parts(
            weights_packed.as_ptr() as *const BlockQ5_Kx8,
            n_row_groups * n_blocks,
        )
    };

    // Warm up
    for _ in 0..10 {
        unsafe {
            gemv_q5_k_transposed(
                weights_bytes,
                stride,
                &input_q8k,
                &mut output_std,
                out_features,
                in_features,
                1,
            );
            gemv_q5_k_x8_neon(
                weights_x8,
                &input_q8k,
                &mut output_x8,
                out_features,
                in_features,
            );
        }
    }

    // Benchmark standard GEMV
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            gemv_q5_k_transposed(
                weights_bytes,
                stride,
                &input_q8k,
                &mut output_std,
                out_features,
                in_features,
                1,
            );
        }
    }
    let std_time = start.elapsed();
    let std_avg = std_time.as_micros() as f64 / iterations as f64;

    // Benchmark interleaved GEMV
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            gemv_q5_k_x8_neon(
                weights_x8,
                &input_q8k,
                &mut output_x8,
                out_features,
                in_features,
            );
        }
    }
    let x8_time = start.elapsed();
    let x8_avg = x8_time.as_micros() as f64 / iterations as f64;

    // Verify correctness
    let max_diff = (0..out_features.min(8))
        .map(|i| (output_std[i] - output_x8[i]).abs())
        .fold(0.0f32, f32::max);

    println!("\nResults:");
    println!("  Standard GEMV:   {:.2} us", std_avg);
    println!("  Interleaved GEMV: {:.2} us", x8_avg);
    println!("  Speedup: {:.2}x", std_avg / x8_avg);
    println!("  Max diff (first 8): {:.6}", max_diff);

    // Performance metrics
    let flops = 2.0 * in_features as f64 * out_features as f64;
    let bytes_read = weights.len() as f64
        * std::mem::size_of::<BlockQ5_K>() as f64
        + in_features as f64;

    println!("\nPerformance:");
    println!("  Standard GFLOPS:   {:.2}", flops / std_avg / 1000.0);
    println!("  Interleaved GFLOPS: {:.2}", flops / x8_avg / 1000.0);
    println!(
        "  Memory bandwidth (std):   {:.2} GB/s",
        bytes_read / std_avg / 1000.0
    );
    println!(
        "  Memory bandwidth (x8):    {:.2} GB/s",
        bytes_read / x8_avg / 1000.0
    );
}
