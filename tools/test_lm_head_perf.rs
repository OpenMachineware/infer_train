// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test lm_head (output projection) GEMV performance

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::model::llama::LlamaModel;
#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::Device;

fn main() {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let args: Vec<String> = std::env::args().collect();
        if args.len() < 2 {
            eprintln!("Usage: {} <model.gguf>", args[0]);
            std::process::exit(1);
        }

        let model_path = &args[1];
        println!("=== lm_head GEMV Performance Test ===");
        println!("Model: {}\n", model_path);

        let ctx = infer_train::inference::InferenceContext::from_gguf(
            model_path,
            Device::Gpu,
        ).expect("Failed to load model");

        let model = &ctx.model;
        let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");

        let hidden_size = model.config.hidden_size;
        let vocab_size = model.output.as_ref().map(|o| o.shape[0]).unwrap_or(hidden_size);

        println!("Hidden size: {}", hidden_size);
        println!("Vocab size: {}", vocab_size);

        // Create input tensor (simulating final norm output)
        let input: Vec<f32> = (0..hidden_size).map(|i| i as f32 * 0.001).collect();

        // Test lm_head GEMV
        println!("\nTest: lm_head GEMV");
        let iterations = 100;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            // This would normally be done by the DAG executor
            // Here we test just the GEMV operation
            let logits = model.forward_gemv_output(&input, gpu_ctx);
        }
        let time = start.elapsed().as_micros() as f64 / iterations as f64;
        println!("  Time: {:.2} μs", time);
        println!("  Throughput: {:.1} M elements/s", (hidden_size * vocab_size) as f64 / time);

        println!("\n=== Summary ===");
        println!("lm_head GEMV: {:.2} μs", time);
        println!("If this is > 100 μs, investigate further");
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("This benchmark requires Metal support");
}
