// Benchmark prefill GEMM performance
use infer_train::backend::arm::gemv::mul_mat_q4_k;
use infer_train::types::{BlockQ4_K, BlockQ8_K, QK_K};
use std::time::Instant;

fn zero_q8k() -> BlockQ8_K {
    BlockQ8_K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }
}

fn main() {
    let out_features = 4096;
    let in_features = 4096;
    let nb = in_features / QK_K;

    println!("Prefill GEMM Benchmark (block-tiling)");
    println!("out_features: {}, in_features: {}", out_features, in_features);

    let weights =
        vec![0u8; out_features * nb * std::mem::size_of::<BlockQ4_K>()];

    // Test different batch sizes
    for n_tokens in [16, 32, 64, 128] {
        let input = vec![zero_q8k(); n_tokens * nb];
        let mut output = vec![0.0f32; n_tokens * out_features];

        // Warmup
        for _ in 0..10 {
            mul_mat_q4_k(
                &weights,
                0,
                &input,
                &mut output,
                out_features,
                in_features,
                n_tokens,
                8,
            );
        }

        // Benchmark
        let iterations = 50;
        let start = Instant::now();
        for _ in 0..iterations {
            mul_mat_q4_k(
                &weights,
                0,
                &input,
                &mut output,
                out_features,
                in_features,
                n_tokens,
                8,
            );
        }
        let elapsed = start.elapsed();
        let per_iter_ms =
            elapsed.as_nanos() as f64 / (iterations as f64) / 1_000_000.0;

        // Calculate GFLOPS
        let flops =
            2.0 * out_features as f64 * in_features as f64 * n_tokens as f64;
        let gflops = flops / (per_iter_ms * 1_000_000.0);

        println!(
            "  n_tokens={}: {:.2} ms, {:.1} GFLOPS",
            n_tokens, per_iter_ms, gflops
        );
    }
}
