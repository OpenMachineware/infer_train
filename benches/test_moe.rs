use infer_train::ops::moe::mul_mat_id_f32;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let n_tokens = 8;
    let n_experts = 8;
    let n_out = 512;
    let n_in = 1024;

    let weights: Vec<f32> = (0..n_experts * n_out * n_in)
        .map(|i| (i % 100) as f32 * 0.01)
        .collect();
    let expert_ids: Vec<i32> =
        (0..n_tokens).map(|i| (i % n_experts) as i32).collect();
    let input: Vec<f32> =
        (0..n_tokens * n_in).map(|i| (i % 50) as f32 * 0.1).collect();
    let mut output = vec![0.0f32; n_tokens * n_out];

    // Warmup
    for _ in 0..10 {
        mul_mat_id_f32(
            &weights,
            n_experts,
            n_out,
            n_in,
            &expert_ids,
            &input,
            &mut output,
        );
        black_box(&output);
    }

    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        mul_mat_id_f32(
            &weights,
            n_experts,
            n_out,
            n_in,
            &expert_ids,
            &input,
            &mut output,
        );
        black_box(&output);
    }
    let total_us = start.elapsed().as_secs_f64() * 1e6;
    println!("Rust: {:.2} us per iteration", total_us / iterations as f64);

    // Calculate GFLOPS
    let ops = n_tokens as f64 * n_out as f64 * n_in as f64 * 2.0; // multiply-add = 2 ops
    let gflops = ops * iterations as f64 / (total_us * 1e-6) / 1e9;
    println!("GFLOPS: {:.2}", gflops);
}
