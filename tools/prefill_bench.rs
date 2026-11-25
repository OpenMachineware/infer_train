// Benchmark prefill GEMM performance with quantization
use infer_train::backend::arm::{gemv, linear, quant};
use infer_train::types::{BlockQ4_K, BlockQ8_K, QK_K};
use std::time::Instant;

fn zero_q8k() -> BlockQ8_K {
    BlockQ8_K { d: 0.0, qs: [0i8; 256], bsums: [0i16; 16] }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let test_full = args.iter().any(|a| a == "--full");

    let out_features = 4096;
    let in_features = 4096;
    let nb = in_features / QK_K;

    println!(
        "Prefill GEMM Benchmark (block-tiling with parallel quantization)"
    );
    println!("out_features: {}, in_features: {}", out_features, in_features);

    let weights =
        vec![0u8; out_features * nb * std::mem::size_of::<BlockQ4_K>()];

    // Test different batch sizes
    let token_sizes = if test_full {
        vec![16, 32, 64, 128, 256, 512]
    } else {
        vec![16, 32, 64, 128]
    };

    for n_tokens in token_sizes {
        // Test 1: Pre-quantized input (GEMM only)
        let input_q8k = vec![zero_q8k(); n_tokens * nb];
        let mut output = vec![0.0f32; n_tokens * out_features];

        // Warmup
        for _ in 0..10 {
            gemv::mul_mat_q4_k(
                &weights,
                0,
                &input_q8k,
                &mut output,
                out_features,
                in_features,
                n_tokens,
                8,
            );
        }

        // Benchmark GEMM only
        let iterations = 50;
        let start = Instant::now();
        for _ in 0..iterations {
            gemv::mul_mat_q4_k(
                &weights,
                0,
                &input_q8k,
                &mut output,
                out_features,
                in_features,
                n_tokens,
                8,
            );
        }
        let elapsed = start.elapsed();
        let gemm_ms =
            elapsed.as_nanos() as f64 / (iterations as f64) / 1_000_000.0;

        // Test 2: Full pipeline with quantization
        let input_f32 = vec![1.0f32; n_tokens * in_features];
        let mut quant_buf = vec![zero_q8k(); n_tokens * nb];
        let mut output2 = vec![0.0f32; n_tokens * out_features];

        // Warmup
        for _ in 0..5 {
            linear::linear_f32_batch(
                &input_f32,
                &weights,
                infer_train::types::QuantType::Q4_K,
                0,
                &mut output2,
                out_features,
                in_features,
                n_tokens,
                8,
                &mut Vec::new(),
                &mut quant_buf,
            )
            .unwrap();
        }

        // Benchmark full pipeline
        let start = Instant::now();
        for _ in 0..iterations {
            linear::linear_f32_batch(
                &input_f32,
                &weights,
                infer_train::types::QuantType::Q4_K,
                0,
                &mut output2,
                out_features,
                in_features,
                n_tokens,
                8,
                &mut Vec::new(),
                &mut quant_buf,
            )
            .unwrap();
        }
        let elapsed = start.elapsed();
        let full_ms =
            elapsed.as_nanos() as f64 / (iterations as f64) / 1_000_000.0;

        // Calculate GFLOPS
        let flops =
            2.0 * out_features as f64 * in_features as f64 * n_tokens as f64;
        let gemm_gflops = flops / (gemm_ms * 1_000_000.0);
        let full_gflops = flops / (full_ms * 1_000_000.0);

        println!(
            "  n_tokens={}: GEMM={:.2}ms ({:.1} GFLOPS), Full={:.2}ms ({:.1} GFLOPS)",
            n_tokens, gemm_ms, gemm_gflops, full_ms, full_gflops
        );
    }
}
