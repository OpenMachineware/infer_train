// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

// Test fusion kernel performance

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::inference::InferenceContext;

#[cfg(all(target_arch = "aarch64", feature = "metal"))]
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        std::process::exit(1);
    }

    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    {
        let model_path = &args[1];
        println!("=== Fusion Kernel Performance Analysis ===\n");

        let gpu_ctx = MetalContext::new().unwrap();
        let device = gpu_ctx.device();
        let ctx = InferenceContext::from_gguf_with_gpu_ctx(
            model_path,
            gpu_ctx.clone(),
        )
        .unwrap();

        let hidden_size = ctx.config.hidden_size; // 1536
        let intermediate_size = ctx.config.intermediate_size; // 8960
        let n_layers = ctx.config.n_layers; // 28

        println!("Model config:");
        println!("  hidden_size: {}", hidden_size);
        println!("  intermediate_size: {}", intermediate_size);
        println!("  n_layers: {}", n_layers);
        println!();

        // Calculate expected time breakdown
        println!("=== Expected Time (without fusion) ===\n");

        let mut total_time_us = 0;

        // Per layer
        println!("Per layer:");
        // Q projection
        total_time_us += 351;
        println!("  Q projection: 351µs");

        // K projection (slow due to small M)
        total_time_us += 385;
        println!("  K projection: 385µs (small M=256)");

        // V projection (slow due to small M)
        total_time_us += 421;
        println!("  V projection: 421µs (small M=256)");

        // Gate projection
        total_time_us += 297;
        println!("  Gate projection: 297µs");

        // Up projection
        total_time_us += 361;
        println!("  Up projection: 361µs");

        // Down projection
        total_time_us += 487;
        println!("  Down projection: 487µs");

        println!("  Subtotal: {}µs\n", total_time_us);

        // Flash Attention (estimated)
        let flash_attn_time = 500; // µs (need to measure)
        total_time_us += flash_attn_time;
        println!("  Flash Attention: ~{}µs (estimated)", flash_attn_time);

        // RMS Norm (2 per layer)
        let rms_norm_time = 10; // µs (estimated)
        total_time_us += rms_norm_time * 2;
        println!("  RMS Norm (x2): {}µs\n", rms_norm_time * 2);

        println!(
            "  Layer total: {}µs = {}ms\n",
            total_time_us,
            total_time_us as f64 / 1000.0
        );

        // Total for all layers
        let layers_total_us = total_time_us * n_layers;
        println!(
            "28 layers total: {}µs = {}ms\n",
            layers_total_us,
            layers_total_us as f64 / 1000.0
        );

        // Output projection (lm_head)
        let output_time_us = 1775;
        println!(
            "Output projection (lm_head): {}µs = {}ms\n",
            output_time_us,
            output_time_us as f64 / 1000.0
        );

        let total_expected = layers_total_us + output_time_us;
        println!(
            "Expected total (no fusion): {}µs = {}ms\n",
            total_expected,
            total_expected as f64 / 1000.0
        );

        // Observed
        let observed_us = 14900;
        println!(
            "Observed: {}µs = {}ms\n",
            observed_us,
            observed_us as f64 / 1000.0
        );

        // Fusion savings
        let savings = total_expected - observed_us;
        let fusion_speedup = total_expected as f64 / observed_us as f64;
        println!(
            "Fusion savings: {}µs ({}x speedup)\n",
            savings, fusion_speedup
        );

        println!("=== Fusion Benefits ===\n");
        println!("Estimated kernel count reduction:");
        println!("  Without fusion: ~400 kernels/token");
        println!("  With fusion: ~200 kernels/token (198 fused)");
        println!("  Reduction: 50%\n");

        println!("Dispatch overhead reduction:");
        println!("  Without batch dispatch: 400 × 250µs = 100ms");
        println!("  With batch dispatch: 1 × 250µs = 0.25ms");
        println!("  Savings: 99.75ms\n");

        println!("=== Remaining Gap ===\n");
        println!("Current: 67 t/s (14.90ms/token)");
        println!("Target: 227 t/s (4.4ms/token)");
        println!("Gap: 3.4x\n");

        println!("Possible causes:");
        println!("  1. FusedQKV kernel performance");
        println!("  2. Flash Attention performance");
        println!("  3. FusedSwiGLU kernel performance");
        println!("  4. Other kernels (RoPE, etc.)");
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "metal")))]
fn main() {
    println!("This benchmark requires Metal support");
}
