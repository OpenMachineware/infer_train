// Benchmark GPU RoPE vs llama.cpp
// Run: cargo run --release --bin benchmark_rope_gpu

use infer_train::ops::gpu::metal::RopeMetalContext;
use infer_train::ops::rope::rope_neox_simple;

fn main() {
    println!("=== GPU RoPE Benchmark ===\n");

    let gpu_ctx = match RopeMetalContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to create GPU context: {}", e);
            return;
        }
    };

    // Test dimensions matching LLaMA
    let configs: Vec<(usize, usize, usize)> = vec![
        (32, 128, 128),  // LLaMA-7B: 32 heads, 128 dim
        (32, 256, 128),  // LLaMA-13B: 40 heads, 128 dim
        (64, 128, 128),  // LLaMA-70B: 64 heads, 128 dim
    ];

    let iterations = 100;
    let freq_base = 10000.0f32;

    println!("{:<15} {:>12} {:>12} {:>8}", "Config", "GPU (us)", "CPU (us)", "Speedup");
    println!("{}", "-".repeat(55));

    for &(n_heads, hidden_dim, n_dims) in &configs {
        let n_seqs = 128;
        let total_size = n_seqs * n_heads * hidden_dim;

        let src: Vec<f32> = (0..total_size).map(|i| (i as f32).sin() / 100.0).collect();
        let positions: Vec<i32> = (0..n_seqs as i32).collect();

        // Warmup GPU
        let _ = gpu_ctx.rope_neox_f32(&src, &positions, n_heads, hidden_dim, n_dims, freq_base);

        // Benchmark GPU
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let _ = gpu_ctx.rope_neox_f32(&src, &positions, n_heads, hidden_dim, n_dims, freq_base);
        }
        let t_gpu = start.elapsed().as_secs_f64() / iterations as f64;

        // Benchmark CPU
        let mut dst_cpu = vec![0.0f32; hidden_dim];
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            for seq in 0..n_seqs {
                for head in 0..n_heads {
                    let offset = seq * n_heads * hidden_dim + head * hidden_dim;
                    rope_neox_simple(&src[offset..], &mut dst_cpu, seq, n_dims, freq_base);
                }
            }
        }
        let t_cpu = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = t_cpu / t_gpu;
        println!("{:<15} {:>12.2} {:>12.2}   {:>7.1}x",
            format!("{}h{}d", n_heads, hidden_dim),
            t_gpu * 1e6, t_cpu * 1e6, speedup);
    }

    // Correctness check
    println!("\n=== Correctness ===");
    let n_heads = 32;
    let hidden_dim = 128;
    let n_dims = 128;
    let n_seqs = 4;

    let total_size = n_seqs * n_heads * hidden_dim;
    let src: Vec<f32> = (0..total_size).map(|i| (i as f32 + 1.0) / 100.0).collect();
    let positions: Vec<i32> = vec![0, 1, 10, 100];

    let dst_gpu = gpu_ctx.rope_neox_f32(&src, &positions, n_heads, hidden_dim, n_dims, freq_base).unwrap();

    let mut dst_cpu = vec![0.0f32; hidden_dim];
    let mut max_diff = 0.0f32;

    for seq in 0..n_seqs {
        for head in 0..n_heads {
            let offset = seq * n_heads * hidden_dim + head * hidden_dim;
            rope_neox_simple(&src[offset..], &mut dst_cpu, positions[seq] as usize, n_dims, freq_base);

            for i in 0..n_dims {
                let diff = (dst_gpu[offset + i] - dst_cpu[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
    }

    println!("max_diff = {:.2e}", max_diff);
}
