// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Layer-by-layer performance breakdown

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
        println!("=== Layer Performance Breakdown ===");
        println!("Model: {}\n", model_path);

        let ctx = infer_train::inference::InferenceContext::from_gguf(
            model_path,
            Device::Gpu,
        ).expect("Failed to load model");

        let model = &ctx.model;
        let gpu_ctx = ctx.gpu_ctx.as_ref().expect("No GPU context");

        let hidden_size = model.config.hidden_size;
        let n_layers = model.layers.len();

        println!("Config: hidden_size={}, n_layers={}\n", hidden_size, n_layers);

        // Test individual operations per layer
        println!("Per-layer operation count:");

        // Attention layer
        println!("  Attention:");
        println!("    1x RMS Norm (attn_norm)");
        println!("    3x GEMV (Q, K, V projections) OR 1x FusedQKV");
        println!("    1x RoPE (Q)");
        println!("    1x RoPE (K)");
        println!("    1x Flash Attention");
        println!("    1x KV Update (K cache)");
        println!("    1x KV Update (V cache)");
        println!("    1x GEMV (output projection)");
        println!("    1x Add (residual)");
        println!("  Total: ~10 kernels per attention layer");

        // FFN layer
        println!("\n  FFN:");
        println!("    1x Add+RMSNorm (ffn_norm) OR 1x FusedAddNorm");
        println!("    2x GEMV (gate, up) OR 1x FusedGateUp");
        println!("    1x SiLU + 1x Mul OR 1x FusedSwiGLU");
        println!("    1x GEMV (down projection)");
        println!("    1x Add (residual)");
        println!("  Total: ~6 kernels per FFN layer (without fusion)");
        println!("         ~3 kernels per FFN layer (with full fusion)");

        println!("\n  Total per layer:");
        println!("    Without fusion: ~16 kernels");
        println!("    With current fusion: ~12 kernels");
        println!("    Optimal (all fusion): ~5-6 kernels");

        println!("\nTotal for {} layers:", n_layers);
        println!("  Without fusion: ~{} kernels", n_layers * 16);
        println!("  With current fusion: ~{} kernels", n_layers * 12);
        println!("  Optimal: ~{} kernels", n_layers * 6);

        // Estimate time
        println!("\n=== Time Estimate (based on ~250 μs dispatch overhead) ===");
        let overhead_per_kernel_us = 250.0;
        let n_kernels_current = n_layers * 12;
        let n_kernels_optimal = n_layers * 6;

        let total_overhead_current = n_kernels_current as f64 * overhead_per_kernel_us / 1000.0;
        let total_overhead_optimal = n_kernels_optimal as f64 * overhead_per_kernel_us / 1000.0;

        println!("Current: {} kernels × 250 μs = {:.1} ms dispatch overhead",
                 n_kernels_current, total_overhead_current);
        println!("Optimal: {} kernels × 250 μs = {:.1} ms dispatch overhead",
                 n_kernels_optimal, total_overhead_optimal);
        println!("Savings: {:.1} ms ({:.0}%)",
                 total_overhead_current - total_overhead_optimal,
                 (total_overhead_current - total_overhead_optimal) / total_overhead_current * 100.0);
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
    println!("This benchmark requires Metal support");
}
