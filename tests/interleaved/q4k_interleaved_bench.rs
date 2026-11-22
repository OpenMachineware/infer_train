use infer_train::backend::arm::gemv::{
    gemv_q4_k_transposed, gemv_q4_k_x8_neon,
};
use infer_train::backend::arm::repack::repack_q4_k_to_x8;
use infer_train::types::{BlockQ4_K, BlockQ4_Kx8, BlockQ8_K, QK_K};
use std::time::Instant;

fn main() {
    println!("=== Q4_K GEMV Benchmark: Standard vs Interleaved ===\n");

    // Test with typical dimensions
    let in_features = 4096;
    let out_features = 4096;
    let n_blocks = in_features / QK_K;

    // Create weights
    let weights: Vec<BlockQ4_K> = (0..out_features * n_blocks)
        .map(|i| BlockQ4_K {
            d: half::f16::from_f32(0.1).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [32; 12],
            qs: [((i % 256) + 1) as u8; 128],
        })
        .collect();

    // Create input
    let input: Vec<BlockQ8_K> = (0..n_blocks)
        .map(|i| BlockQ8_K {
            d: 1.0 + (i % 10) as f32 * 0.1,
            qs: [((i % 128) + 1) as i8; 256],
            bsums: [16; 16],
        })
        .collect();

    let mut output_std = vec![0.0f32; out_features];
    let mut output_x8 = vec![0.0f32; out_features];

    let stride = n_blocks * std::mem::size_of::<BlockQ4_K>();
    let weights_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * std::mem::size_of::<BlockQ4_K>(),
        )
    };

    // Repack weights
    println!("Repacking weights...");
    let start = Instant::now();
    let n_row_groups = out_features / 8;
    let mut weights_packed: Vec<u8> =
        vec![0u8; n_row_groups * n_blocks * std::mem::size_of::<BlockQ4_Kx8>()];
    repack_q4_k_to_x8(
        weights_bytes,
        stride,
        &mut weights_packed,
        out_features,
        n_blocks,
    );
    println!("Repack time: {:.2?}", start.elapsed());

    let weights_x8: &[BlockQ4_Kx8] = unsafe {
        std::slice::from_raw_parts(
            weights_packed.as_ptr() as *const BlockQ4_Kx8,
            n_row_groups * n_blocks,
        )
    };

    // Warmup
    for _ in 0..10 {
        unsafe {
            gemv_q4_k_transposed(
                weights_bytes,
                stride,
                &input,
                &mut output_std,
                out_features,
                in_features,
                1,
            );
            gemv_q4_k_x8_neon(
                weights_x8,
                &input,
                &mut output_x8,
                out_features,
                in_features,
            );
        }
    }

    // Benchmark standard GEMV
    let iters = 50;
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            gemv_q4_k_transposed(
                weights_bytes,
                stride,
                &input,
                &mut output_std,
                out_features,
                in_features,
                1,
            );
        }
    }
    let std_time = start.elapsed();
    let std_avg = std_time.as_micros() as f64 / iters as f64;

    // Benchmark interleaved GEMV
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            gemv_q4_k_x8_neon(
                weights_x8,
                &input,
                &mut output_x8,
                out_features,
                in_features,
            );
        }
    }
    let x8_time = start.elapsed();
    let x8_avg = x8_time.as_micros() as f64 / iters as f64;

    // Verify correctness
    let max_diff = (0..out_features.min(8))
        .map(|i| (output_std[i] - output_x8[i]).abs())
        .fold(0.0f32, f32::max);

    let flops = 2 * out_features * in_features;
    let total_bytes = weights.len() * std::mem::size_of::<BlockQ4_K>()
        + input.len() * std::mem::size_of::<BlockQ8_K>()
        + out_features * 4;

    println!("\n=== Results ===");
    println!(
        "Standard GEMV:   {:.2} us ({:.1} GFLOPS, {:.1} GB/s)",
        std_avg,
        flops as f64 / std_avg / 1000.0,
        total_bytes as f64 / std_avg / 1000.0
    );
    println!(
        "Interleaved GEMV: {:.2} us ({:.1} GFLOPS, {:.1} GB/s)",
        x8_avg,
        flops as f64 / x8_avg / 1000.0,
        total_bytes as f64 / x8_avg / 1000.0
    );
    println!("Speedup: {:.2}x", std_avg / x8_avg);
    println!("Max diff: {:.6}", max_diff);
}
