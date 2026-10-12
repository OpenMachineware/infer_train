use half::f16;
use infer_train::gpu::MetalContext;
use infer_train::ops::attention_fused::flash_attention_decode;
use std::time::Instant;

fn main() {
    let ctx = MetalContext::new().unwrap();

    // Small test: seq_k=256, F16
    let num_heads = 32;
    let head_dim = 128;
    let seq_k = 256;

    // Initialize random data
    let mut q_data = vec![f16::from_f32(0.5); num_heads * head_dim];
    let mut k_data = vec![f16::from_f32(0.3); head_dim * seq_k];
    let mut v_data = vec![f16::from_f32(0.2); head_dim * seq_k];

    // Add some variation
    for i in 0..q_data.len() {
        q_data[i] = f16::from_f32(0.1 * (i as f32 % 10.0 - 5.0));
    }
    for i in 0..k_data.len() {
        k_data[i] = f16::from_f32(0.05 * (i as f32 % 20.0 - 10.0));
    }
    for i in 0..v_data.len() {
        v_data[i] = f16::from_f32(0.03 * (i as f32 % 15.0 - 7.5));
    }

    let output = flash_attention_decode(
        &ctx, &q_data, &k_data, &v_data, num_heads, head_dim, seq_k, 1.0, None,
    )
    .unwrap();

    // Warmup
    for _ in 0..10 {
        let _ = flash_attention_decode(
            &ctx, &q_data, &k_data, &v_data, num_heads, head_dim, seq_k, 1.0,
            None,
        )
        .unwrap();
    }

    // Benchmark
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = flash_attention_decode(
            &ctx, &q_data, &k_data, &v_data, num_heads, head_dim, seq_k, 1.0,
            None,
        )
        .unwrap();
    }
    let elapsed = start.elapsed();
    let us_per_iter = elapsed.as_micros() as f64 / iterations as f64;

    // GFLOPS = 2 * num_heads * seq_k * head_dim / (us * 1000)
    let flops = 2.0 * (num_heads as f64) * (seq_k as f64) * (head_dim as f64);
    let gflops = flops / (us_per_iter * 1000.0);

    println!(
        "F16, seq_k={}: {:.2}µs ({:.2} GFLOPS)",
        seq_k, us_per_iter, gflops
    );
    println!(
        "Output checksum: {:.6}",
        output.iter().map(|x| x.to_f32()).sum::<f32>()
    );
}
