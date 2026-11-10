// Direct test of interleaved GEMV vs standard GEMV
use infer_train::quant::types::{BlockQ4K, BlockQ8K, QK_K};
use infer_train::quant::vec_dot::arm::k_quant_batch::{
    BlockQ4Kx8, gemv_q4_k_interleaved, repack_q4_k_to_interleaved,
};
use infer_train::quant::vec_dot::arm::vec_dot_q4_k_q8_k_neon;
use std::time::Instant;

fn main() {
    println!("=== Interleaved vs Standard GEMV Direct Test ===\n");

    // Test with realistic dimensions (Qwen3-0.6B-like)
    let k = 4096;
    let m = 4096; // Output dimension (must be multiple of 8 for interleaved)

    let nb = k / QK_K;

    // Generate test weights
    let weights: Vec<BlockQ4K> =
        (0..(m * nb)).map(|_| unsafe { std::mem::zeroed() }).collect();

    // Generate test input
    let input: Vec<BlockQ8K> =
        (0..nb).map(|_| unsafe { std::mem::zeroed() }).collect();

    let mut output_std = vec![0.0f32; m];
    let mut output_inter = vec![0.0f32; m];

    // Test standard GEMV (simple vec_dot loop)
    let iterations = 50;
    let start = Instant::now();
    for _ in 0..iterations {
        for row in 0..m {
            unsafe {
                output_std[row] = vec_dot_q4_k_q8_k_neon(
                    k,
                    &weights[row * nb..(row + 1) * nb],
                    &input,
                );
            }
        }
    }
    let std_time = start.elapsed() / iterations;
    let std_gbs = (m as f64 * k as f64 * 4.0 / 1e9) / std_time.as_secs_f64();
    println!("Standard GEMV (vec_dot loop): {:?}", std_time);
    println!("  Bandwidth: {:.2} GB/s", std_gbs);

    // Repack weights to interleaved format
    println!("\nRepacking weights to interleaved format...");
    let n_interleaved = m / 8;
    let mut interleaved_weights: Vec<BlockQ4Kx8> =
        Vec::with_capacity(n_interleaved * nb);

    let repack_start = Instant::now();
    for row_group in (0..m).step_by(8) {
        for block_idx in 0..nb {
            let mut input_blocks: [BlockQ4K; 8] = unsafe { std::mem::zeroed() };
            for r in 0..8 {
                input_blocks[r] = weights[(row_group + r) * nb + block_idx];
            }
            interleaved_weights
                .push(unsafe { repack_q4_k_to_interleaved(&input_blocks) });
        }
    }
    let repack_time = repack_start.elapsed();
    println!("  Repack time: {:?}", repack_time);
    println!(
        "  Interleaved blocks: {} (from {} original)",
        interleaved_weights.len(),
        m * nb
    );

    // Test interleaved GEMV
    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            gemv_q4_k_interleaved(
                &interleaved_weights,
                &input,
                &mut output_inter,
                k,
                m,
            );
        }
    }
    let inter_time = start.elapsed() / iterations;
    let inter_gbs =
        (m as f64 * k as f64 * 4.0 / 1e9) / inter_time.as_secs_f64();
    println!("\nInterleaved GEMV: {:?}", inter_time);
    println!("  Bandwidth: {:.2} GB/s", inter_gbs);

    // Calculate improvement
    let speedup = std_time.as_secs_f64() / inter_time.as_secs_f64();
    println!("\n=== Result ===");
    println!("Speedup: {:.2}x", speedup);
    println!(
        "Interleaved is {} faster",
        if speedup > 1.0 {
            format!("{:.1}%", (speedup - 1.0) * 100.0)
        } else {
            format!("slower by {:.1}%", (1.0 - speedup) * 100.0)
        }
    );

    // Verify correctness (should be equal)
    let mut max_error = 0.0f32;
    for i in 0..m {
        let error = (output_std[i] - output_inter[i]).abs();
        if error > max_error {
            max_error = error;
        }
    }
    println!("\nMax error: {}", max_error);
}
