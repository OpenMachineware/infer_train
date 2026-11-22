use infer_train::backend::arm::gemv::*;
use infer_train::backend::arm::repack::repack_q5_k_to_x8;
use infer_train::types::*;
use std::time::Instant;

fn main() {
    println!("=== Q5_K GEMV Benchmark: Standard vs Interleaved ===\n");

    // Q projection: in_features=1024, out_features=2048
    let in_features = 1024;
    let out_features = 2048;
    let n_blocks = in_features / 256; // 4
    let block_size = std::mem::size_of::<BlockQ5_K>(); // 176

    println!("in_features: {}, out_features: {}", in_features, out_features);
    println!("n_blocks: {}, block_size: {} bytes", n_blocks, block_size);

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

    let mut output_std = vec![0.0f32; out_features];
    let mut output_x8 = vec![0.0f32; out_features];

    let weights_bytes = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * block_size,
        )
    };
    let col_stride = n_blocks * block_size;

    // Repack weights for interleaved GEMV
    println!("Repacking weights...");
    let start_repack = Instant::now();
    let n_row_groups = out_features / 8;
    let mut weights_packed: Vec<u8> =
        vec![0u8; n_row_groups * n_blocks * std::mem::size_of::<BlockQ5_Kx8>()];
    repack_q5_k_to_x8(
        weights_bytes,
        col_stride,
        &mut weights_packed,
        out_features,
        n_blocks,
    );
    println!("Repack time: {:.2?}", start_repack.elapsed());

    let weights_x8: &[BlockQ5_Kx8] = unsafe {
        std::slice::from_raw_parts(
            weights_packed.as_ptr() as *const BlockQ5_Kx8,
            n_row_groups * n_blocks,
        )
    };

    // Warmup
    for _ in 0..10 {
        unsafe {
            gemv_q5_k_transposed(
                weights_bytes,
                col_stride,
                &input,
                &mut output_std,
                out_features,
                in_features,
                1,
            );
            gemv_q5_k_x8_neon(
                weights_x8,
                &input,
                &mut output_x8,
                out_features,
                in_features,
            );
        }
    }

    // Benchmark standard GEMV
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            gemv_q5_k_transposed(
                weights_bytes,
                col_stride,
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
            gemv_q5_k_x8_neon(
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
    let weight_bytes = weights.len() * block_size;
    let input_bytes = input.len() * std::mem::size_of::<BlockQ8_K>();
    let output_bytes = output_std.len() * 4;
    let total_bytes = weight_bytes + input_bytes + output_bytes;

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

    // Test with larger dimension
    println!("\n=== Larger dimension test (4096x4096) ===");
    let in_features2 = 4096;
    let out_features2 = 4096;
    let n_blocks2 = in_features2 / 256;
    let col_stride2 = n_blocks2 * block_size;

    let weights2: Vec<BlockQ5_K> = (0..out_features2 * n_blocks2)
        .map(|i| BlockQ5_K {
            d: half::f16::from_f32(0.1).to_bits(),
            dmin: half::f16::from_f32(0.0).to_bits(),
            scales: [32; 12],
            qh: [0; 32],
            qs: [128; 128],
        })
        .collect();

    let input2: Vec<BlockQ8_K> =
        vec![BlockQ8_K { d: 1.0, qs: [1; 256], bsums: [0; 16] }; n_blocks2];

    let mut output2_std = vec![0.0f32; out_features2];
    let mut output2_x8 = vec![0.0f32; out_features2];

    let weights2_bytes = unsafe {
        std::slice::from_raw_parts(
            weights2.as_ptr() as *const u8,
            weights2.len() * block_size,
        )
    };

    // Repack
    let n_row_groups2 = out_features2 / 8;
    let mut weights2_packed: Vec<u8> =
        vec![
            0u8;
            n_row_groups2 * n_blocks2 * std::mem::size_of::<BlockQ5_Kx8>()
        ];
    repack_q5_k_to_x8(
        weights2_bytes,
        col_stride2,
        &mut weights2_packed,
        out_features2,
        n_blocks2,
    );
    let weights2_x8: &[BlockQ5_Kx8] = unsafe {
        std::slice::from_raw_parts(
            weights2_packed.as_ptr() as *const BlockQ5_Kx8,
            n_row_groups2 * n_blocks2,
        )
    };

    // Warmup
    for _ in 0..5 {
        unsafe {
            gemv_q5_k_transposed(
                weights2_bytes,
                col_stride2,
                &input2,
                &mut output2_std,
                out_features2,
                in_features2,
                1,
            );
            gemv_q5_k_x8_neon(
                weights2_x8,
                &input2,
                &mut output2_x8,
                out_features2,
                in_features2,
            );
        }
    }

    // Benchmark
    let iters2 = 50;
    let start = Instant::now();
    for _ in 0..iters2 {
        unsafe {
            gemv_q5_k_transposed(
                weights2_bytes,
                col_stride2,
                &input2,
                &mut output2_std,
                out_features2,
                in_features2,
                1,
            );
        }
    }
    let std_time2 = start.elapsed();
    let std_avg2 = std_time2.as_micros() as f64 / iters2 as f64;

    let start = Instant::now();
    for _ in 0..iters2 {
        unsafe {
            gemv_q5_k_x8_neon(
                weights2_x8,
                &input2,
                &mut output2_x8,
                out_features2,
                in_features2,
            );
        }
    }
    let x8_time2 = start.elapsed();
    let x8_avg2 = x8_time2.as_micros() as f64 / iters2 as f64;

    let flops2 = 2 * out_features2 * in_features2;
    let total_bytes2 = weights2.len() * block_size
        + input2.len() * std::mem::size_of::<BlockQ8_K>()
        + output2_std.len() * 4;

    println!(
        "Standard GEMV:   {:.2} us ({:.1} GFLOPS, {:.1} GB/s)",
        std_avg2,
        flops2 as f64 / std_avg2 / 1000.0,
        total_bytes2 as f64 / std_avg2 / 1000.0
    );
    println!(
        "Interleaved GEMV: {:.2} us ({:.1} GFLOPS, {:.1} GB/s)",
        x8_avg2,
        flops2 as f64 / x8_avg2 / 1000.0,
        total_bytes2 as f64 / x8_avg2 / 1000.0
    );
    println!("Speedup: {:.2}x", std_avg2 / x8_avg2);
}
