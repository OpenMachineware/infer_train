// Test GEMM performance in isolation
// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        use infer_train::inference::{Device, InferenceContext};
        use infer_train::quant::vec_dot::gpu::metal::MetalContext;

        let model_path = "Qwen3-0.6B-UD-Q4_K_XL.gguf";

        // Load model to get weights
        let cpu_ctx = InferenceContext::from_gguf(model_path, Device::Cpu)
            .expect("Failed to load model");

        let wq = cpu_ctx.model.layers[0].wq.as_ref().expect("No wq");
        println!("WQ: dtype={:?}, shape={:?}", wq.dtype, wq.shape);

        // Create GPU context
        let gpu_ctx =
            MetalContext::new().expect("Failed to create GPU context");

        // Test parameters
        let k = wq.shape[0]; // 1024
        let m = wq.shape[1]; // 2048 (n_heads * head_dim)
        let batch_sizes = vec![1, 8, 32, 128];

        for &batch in &batch_sizes {
            // Create random input
            let x: Vec<f32> =
                (0..batch * k).map(|i| (i as f32 * 0.01).sin()).collect();

            let start = std::time::Instant::now();

            // Run 10 iterations
            for _ in 0..10 {
                let weights: &[infer_train::quant::types::BlockQ4K] =
                    bytemuck::cast_slice(wq.bytes());
                let _output = gpu_ctx
                    .matmul_q4_k_f32(m, batch, k, weights, &x)
                    .expect("GEMM failed");
            }

            let elapsed = start.elapsed();
            let ms_per_iter = elapsed.as_millis() as f64 / 10.0;
            let gflops = (2.0 * batch as f64 * m as f64 * k as f64 * 10.0
                / elapsed.as_secs_f64())
                / 1e9;

            println!(
                "Batch {}: {:.1}ms/iter, {:.1} GFLOPS",
                batch, ms_per_iter, gflops
            );
        }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    {
        println!("Metal not available");
    }
}
