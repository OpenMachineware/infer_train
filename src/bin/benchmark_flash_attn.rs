use infer_train::ops::gpu::FlashAttnMetalContext;
use half::f16;

fn main() {
    println!("=== Flash Attention F16 Benchmark ===\n");

    let ctx = FlashAttnMetalContext::new().expect("Failed to create Metal context");

    // Test parameters
    let head_dim = 128;
    let seq_len = 512;

    // Create test data
    let mut q = vec![f16::ZERO; head_dim];
    let mut k = vec![f16::ZERO; seq_len * head_dim];
    let mut v = vec![f16::ZERO; seq_len * head_dim];

    // Initialize with random values
    for i in 0..head_dim {
        q[i] = f16::from_f32(rand_value());
    }
    for i in 0..seq_len * head_dim {
        k[i] = f16::from_f32(rand_value());
        v[i] = f16::from_f32(rand_value());
    }

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Test decode (batch=1)
    println!("Testing decode (batch=1, head_dim={}, seq_len={}):", head_dim, seq_len);

    let start = std::time::Instant::now();
    let result = ctx.flash_attn_f16_decode(&q, &k, &v, None, head_dim, seq_len, scale)
        .expect("Flash attention failed");
    let elapsed = start.elapsed();

    println!("  Output: {} values", result.len());
    println!("  Time: {:?}", elapsed);

    // Verify output
    let mut non_zero = 0;
    for i in 0..head_dim {
        if result[i] != f16::ZERO {
            non_zero += 1;
        }
    }
    println!("  Non-zero outputs: {}", non_zero);

    // Test multi-head (batch=4, n_heads=32)
    let batch_size = 4;
    let n_heads = 32;
    let n_kv_heads = 32;  // No GQA

    println!("\nTesting multi-head (batch={}, n_heads={}, head_dim={}, seq_len={}):",
        batch_size, n_heads, head_dim, seq_len);

    let q_multi = vec![f16::from_f32(rand_value()); batch_size * n_heads * head_dim];
    let k_multi = vec![f16::from_f32(rand_value()); seq_len * n_kv_heads * head_dim];
    let v_multi = vec![f16::from_f32(rand_value()); seq_len * n_kv_heads * head_dim];

    let start = std::time::Instant::now();
    let result_multi = ctx.flash_attn_f16_multihead(
        &q_multi, &k_multi, &v_multi, None,
        batch_size, n_heads, n_kv_heads, head_dim, seq_len, scale
    ).expect("Flash attention failed");
    let elapsed_multi = start.elapsed();

    println!("  Output: {} values", result_multi.len());
    println!("  Time: {:?}", elapsed_multi);

    // Benchmark across different seq_lens
    println!("\n=== Performance across seq_lens ===");
    for &sl in &[64, 128, 256, 512, 1024, 2048, 4096] {
        let k = vec![f16::from_f32(rand_value()); sl * head_dim];
        let v = vec![f16::from_f32(rand_value()); sl * head_dim];

        let start = std::time::Instant::now();
        let _ = ctx.flash_attn_f16_decode(&q, &k, &v, None, head_dim, sl, scale)
            .expect("Flash attention failed");
        let elapsed = start.elapsed();

        println!("  seq_len={}: {:?}", sl, elapsed);
    }

    println!("\nBenchmark complete!");
}

fn rand_value() -> f32 {
    // Simple pseudo-random for reproducibility
    static mut SEED: u64 = 12345;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32 / u64::MAX as f32) * 2.0 - 1.0
    }
}
