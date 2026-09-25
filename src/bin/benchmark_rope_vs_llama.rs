// Benchmark RoPE: Fair comparison with identical logic
// Run: cargo run --release --bin benchmark_rope_vs_llama

use infer_train::ops::rope::rope_neox_simple;

// llama.cpp's exact scalar (separate cos/sin calls)
fn rope_llama_cpp(src: &[f32], dst: &mut [f32], pos: usize, n_dims: usize, freq_base: f32) {
    let n_dims_half = n_dims / 2;
    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    let mut theta = pos as f32;

    for ic in 0..n_dims_half {
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        dst[ic] = src[ic] * cos_theta - src[ic + n_dims_half] * sin_theta;
        dst[ic + n_dims_half] = src[ic] * sin_theta + src[ic + n_dims_half] * cos_theta;
        theta *= theta_scale;
    }
}

// llama.cpp style with sin_cos (our optimization)
fn rope_llama_cpp_sincos(src: &[f32], dst: &mut [f32], pos: usize, n_dims: usize, freq_base: f32) {
    let n_dims_half = n_dims / 2;
    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    let mut theta = pos as f32;

    for ic in 0..n_dims_half {
        let (sin_theta, cos_theta) = theta.sin_cos();
        dst[ic] = src[ic] * cos_theta - src[ic + n_dims_half] * sin_theta;
        dst[ic + n_dims_half] = src[ic] * sin_theta + src[ic + n_dims_half] * cos_theta;
        theta *= theta_scale;
    }
}

fn benchmark_rope() {
    println!("=== RoPE: Ours vs llama.cpp (separate cos/sin) vs llama.cpp+sin_cos ===\n");

    let head_dims: Vec<usize> = vec![64, 128, 256, 512, 1024];
    let iterations = 1000;
    let freq_base = 10000.0f32;

    println!("Head Dimension Scaling (batch=128, {} iterations):", iterations);
    println!("{:<10} {:>10} {:>10} {:>10} {:>8}", "Dim", "Ours", "llama.cpp", "llama+sc", "vs llama");
    println!("{}", "-".repeat(60));

    let batch = 128usize;
    for &head_dim in &head_dims {
        let total_size = head_dim * batch;
        let src: Vec<f32> = (0..total_size).map(|i| (i as f32).sin() / 100.0).collect();
        let mut dst = vec![0.0f32; total_size];

        // Warmup
        for pos in 0..batch {
            rope_neox_simple(&src[pos * head_dim..], &mut dst[pos * head_dim..], pos, head_dim, freq_base);
            rope_llama_cpp(&src[pos * head_dim..], &mut dst[pos * head_dim..], pos, head_dim, freq_base);
            rope_llama_cpp_sincos(&src[pos * head_dim..], &mut dst[pos * head_dim..], pos, head_dim, freq_base);
        }

        // Benchmark ours (with sin_cos optimization)
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            for pos in 0..batch {
                rope_neox_simple(&src[pos * head_dim..], &mut dst[pos * head_dim..], pos, head_dim, freq_base);
            }
        }
        let t_ours = start.elapsed().as_secs_f64() / iterations as f64;

        // Benchmark llama.cpp (separate cos/sin)
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            for pos in 0..batch {
                rope_llama_cpp(&src[pos * head_dim..], &mut dst[pos * head_dim..], pos, head_dim, freq_base);
            }
        }
        let t_llama = start.elapsed().as_secs_f64() / iterations as f64;

        // Benchmark llama.cpp style with sin_cos
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            for pos in 0..batch {
                rope_llama_cpp_sincos(&src[pos * head_dim..], &mut dst[pos * head_dim..], pos, head_dim, freq_base);
            }
        }
        let t_llama_sc = start.elapsed().as_secs_f64() / iterations as f64;

        let ratio = t_llama / t_ours * 100.0;
        println!("{:<10} {:>10.2} {:>10.2} {:>10.2}   {:>7.1}%",
            head_dim, t_ours * 1e6, t_llama * 1e6, t_llama_sc * 1e6, ratio);
    }

    // Correctness
    println!("\n=== Correctness ===");
    let src: Vec<f32> = (0..128).map(|i| (i as f32 + 1.0) / 100.0).collect();
    let mut dst_ours = vec![0.0f32; 128];
    let mut dst_llama = vec![0.0f32; 128];

    for pos in 0..3 {
        rope_neox_simple(&src, &mut dst_ours, pos, 128, freq_base);
        rope_llama_cpp(&src, &mut dst_llama, pos, 128, freq_base);

        let max_diff = (0..128).map(|i| (dst_ours[i] - dst_llama[i]).abs()).fold(0.0_f32, f32::max);
        println!("Position {}: max_diff = {:.2e}", pos, max_diff);
    }
}

fn main() {
    benchmark_rope();
}
