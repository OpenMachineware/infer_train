// SSM GPU operator performance benchmark
// Compare with llama.cpp implementation
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
mod bench {
    use infer_train::quant::vec_dot::gpu::metal::MetalContext;
    use std::time::Instant;

    fn measure_time<F: FnMut()>(mut f: F, iterations: usize) -> (f64, f64) {
        let mut times = Vec::with_capacity(iterations);

        // Warm up
        for _ in 0..10 {
            f();
        }

        // Measure
        for _ in 0..iterations {
            let start = Instant::now();
            f();
            times.push(start.elapsed().as_nanos() as f64);
        }

        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
            / times.len() as f64;
        let std = variance.sqrt();

        (mean, std)
    }

    fn compute_gflops(
        d_inner: usize,
        n_tok: usize,
        d_conv: usize,
        time_ns: f64,
    ) -> f64 {
        // FLOPs: d_inner * n_tok * (d_conv * 2 + 2)
        // - d_conv multiplications
        // - d_conv - 1 additions
        // - 1 bias addition
        // Total: d_conv * 2 operations
        let flops = d_inner as f64 * n_tok as f64 * (d_conv * 2) as f64;
        flops / time_ns * 1000.0 // GFLOPS
    }

    fn compute_scan_gflops(
        d_inner: usize,
        n_tok: usize,
        d_state: usize,
        time_ns: f64,
    ) -> f64 {
        // FLOPs per token:
        // - softplus: ~10 ops (exp, log, add)
        // - decay: 1 exp, 1 mul
        // - state update: d_state * (1 mul + 1 mul + 1 add)
        // - output: d_state * (1 mul + 1 add)
        // Total per token: ~10 + d_state * 5
        let flops =
            d_inner as f64 * n_tok as f64 * (10.0 + d_state as f64 * 5.0);
        flops / time_ns * 1000.0 // GFLOPS
    }

    /// Benchmark SSM Conv1D
    pub fn bench_ssm_conv1d() {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        // Qwen3.8-27B parameters
        // ssm_conv1d.weight shape: [4, 10240] -> d_conv=4, d_inner=10240
        let d_conv = 4;
        let d_inner = 10240;

        // Test different sequence lengths
        let seq_lengths = [1, 16, 64, 128, 256, 512, 1024, 2048];

        println!(
            "\n=== SSM Conv1D Performance (d_conv={}, d_inner={}) ===\n",
            d_conv, d_inner
        );
        println!("seq_len | time (us) | GFLOPS | GB/s");
        println!("--------|-----------|--------|------");

        for n_tok in seq_lengths {
            let x_size = d_inner * (n_tok + d_conv - 1);
            let w_size = d_inner * d_conv;

            let x = vec![0.1f32; x_size];
            let w = vec![0.05f32; w_size];
            let bias = vec![0.01f32; d_inner];

            let (mean, std) = measure_time(
                || {
                    let _ =
                        ctx.ssm_conv1d_d4(&x, &w, Some(&bias), d_inner, n_tok);
                },
                100,
            );

            let gflops = compute_gflops(d_inner, n_tok, d_conv, mean);

            // Memory bandwidth: read x + w + bias, write y
            let bytes_read = (x_size + w_size + d_inner) * 4;
            let bytes_write = d_inner * n_tok * 4;
            let total_bytes = (bytes_read + bytes_write) as f64;
            let bandwidth_gbps = total_bytes / mean * 1000.0;

            println!(
                "{:7} | {:7.2} ± {:.1} | {:6.2} | {:5.1}",
                n_tok,
                mean / 1000.0,
                std / 1000.0,
                gflops,
                bandwidth_gbps
            );
        }
    }

    /// Benchmark SSM Scan
    pub fn bench_ssm_scan() {
        let ctx = MetalContext::new().expect("Failed to create Metal context");

        // Mamba-1 parameters: d_state=16
        // For Qwen3.8, d_state might be different
        let d_state = 16;
        let d_inner = 4096; // Typical hidden size

        let seq_lengths = [1, 16, 64, 128, 256, 512, 1024, 2048];

        println!(
            "\n=== SSM Scan Performance (d_state={}, d_inner={}) ===\n",
            d_state, d_inner
        );
        println!("seq_len | time (us) | GFLOPS | GB/s");
        println!("--------|-----------|--------|------");

        for n_tok in seq_lengths {
            let x_size = d_inner * n_tok;
            let a_size = d_inner * d_state;
            let bc_size = d_state * n_tok;

            let x = vec![0.1f32; x_size];
            let dt = vec![0.001f32; x_size];
            let a = vec![-0.1f32; a_size];
            let b = vec![0.05f32; bc_size];
            let c = vec![0.05f32; bc_size];

            let (mean, std) = measure_time(
                || {
                    let _ = ctx.ssm_scan_d16(
                        &x, &dt, &a, &b, &c, None, d_inner, n_tok,
                    );
                },
                100,
            );

            let gflops = compute_scan_gflops(d_inner, n_tok, d_state, mean);

            // Memory bandwidth
            let bytes_read = (x_size + x_size + a_size + bc_size * 2) * 4;
            let bytes_write = (x_size + a_size) * 4;
            let total_bytes = (bytes_read + bytes_write) as f64;
            let bandwidth_gbps = total_bytes / mean * 1000.0;

            println!(
                "{:7} | {:7.2} ± {:.1} | {:6.2} | {:5.1}",
                n_tok,
                mean / 1000.0,
                std / 1000.0,
                gflops,
                bandwidth_gbps
            );
        }
    }

    pub fn run_benchmarks() {
        println!("\n========================================");
        println!("SSM GPU Operator Performance Benchmark");
        println!("========================================");

        bench_ssm_conv1d();
        bench_ssm_scan();

        println!("\n=== Summary ===");
        println!("SSM Conv1D: 1D convolution for Mamba architecture");
        println!("SSM Scan: Recurrent state update");
        println!("\nNext steps:");
        println!("1. Compare with llama.cpp CUDA performance");
        println!("2. Optimize kernels if needed");
        println!("3. Add SIMD/vectorization for better throughput");
    }
}

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        bench::run_benchmarks();
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("This benchmark requires Metal support on aarch64");
    }
}
