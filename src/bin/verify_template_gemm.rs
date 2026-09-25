// Verify templated GEMM kernels from llama.cpp
// Run: cargo run --release --bin verify_template_gemm

use infer_train::quant::types::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K};
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use half::f16;

fn generate_q4_k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    let mut weights = Vec::with_capacity(m * nb);
    for i in 0..(m * nb) {
        let mut scales = [0u8; 12];
        let mut qs = [0u8; 128];
        for j in 0..12 {
            scales[j] = (40 + (j % 8)) as u8;
        }
        for j in 0..128 {
            qs[j] = ((i * 17 + j * 3) % 256) as u8;
        }
        weights.push(BlockQ4K {
            d: f16::from_f32(1.0).to_bits(),
            dmin: f16::from_f32(0.0).to_bits(),
            scales,
            qs,
        });
    }
    weights
}

fn generate_f32_input(size: usize) -> Vec<f32> {
    (0..size).map(|i| ((i % 100) as f32 * 0.01 - 0.5)).collect()
}

fn main() {
    println!("=== Templated GEMM Kernel Verification ===\n");

    let ctx = MetalContext::new().expect("Failed to create MetalContext");

    // Test parameters
    let m = 4096;
    let k = 4096;
    let n = 128;

    let weights = generate_q4_k_weights(m, k);
    let input = generate_f32_input(k * n);

    println!("Testing Q4_K templated GEMM (M={}, N={}, K={})", m, n, k);
    println!("Weights: {} blocks ({} bytes), Input: {} elements", weights.len(), weights.len() * std::mem::size_of::<BlockQ4K>(), input.len());
    
    // Check input layout
    println!("\nInput matrix layout check:");
    println!("  Element 0: {:.4}", input[0]);
    println!("  Element K: {:.4} (expect column 1 start if column-major)", input[k]);
    println!("  Element K+1: {:.4}", input[k+1]);
    
    // Test templated kernel
    println!("\n1. Testing templated kernel...");
    let template_result = match ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input) {
        Ok(result) => {
            println!("   Result size: {} values", result.len());
            println!("   Sample values: {:.4}, {:.4}, {:.4}, {:.4}",
                     result[0], result[100], result[1000], result[10000]);
            println!("   ✓ Templated kernel executed successfully");
            Some(result)
        }
        Err(e) => {
            println!("   ✗ Templated kernel failed: {}", e);
            None
        }
    };

    // Test hand-written kernel for comparison
    println!("\n2. Testing hand-written kernel...");
    let hand_result = match ctx.gemm_q4_k_f32(m, n, k, &weights, &input) {
        Ok(result) => {
            println!("   Result size: {} values", result.len());
            println!("   Sample values: {:.4}, {:.4}, {:.4}, {:.4}",
                     result[0], result[100], result[1000], result[10000]);
            println!("   ✓ Hand-written kernel executed successfully");
            Some(result)
        }
        Err(e) => {
            println!("   ✗ Hand-written kernel failed: {}", e);
            None
        }
    };
    
    // Compare results
    if let (Some(t), Some(h)) = (&template_result, &hand_result) {
        println!("\n3. Result comparison:");
        let ratio = t[0] / h[0];
        println!("   Ratio of first values: {:.4} / {:.4} = {:.4}", t[0], h[0], ratio);
        
        // Check if ratio is consistent
        let ratios: Vec<f64> = t.iter().zip(h.iter())
            .filter(|(_, hv)| hv.abs() > 0.001)
            .take(100)
            .map(|(tv, hv)| *tv as f64 / *hv as f64)
            .collect();
        
        if !ratios.is_empty() {
            let avg_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
            println!("   Average ratio (first 100 non-zero): {:.4}", avg_ratio);
        }
    }

    // Performance comparison
    println!("\n3. Performance comparison (10 iterations)...");

    // Templated
    let start = std::time::Instant::now();
    for _ in 0..10 {
        let _ = ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input).unwrap();
    }
    let templated_time = start.elapsed().as_micros() as f64 / 10.0;

    // Hand-written
    let start = std::time::Instant::now();
    for _ in 0..10 {
        let _ = ctx.gemm_q4_k_f32(m, n, k, &weights, &input).unwrap();
    }
    let handwritten_time = start.elapsed().as_micros() as f64 / 10.0;

    let ops = 2.0 * m as f64 * n as f64 * k as f64;
    let templated_gflops = ops / templated_time / 1000.0;
    let handwritten_gflops = ops / handwritten_time / 1000.0;

    println!("\n┌──────────────────┬──────────┬──────────┐");
    println!("│ Kernel           │ Time (µs)│ GFLOPS   │");
    println!("├──────────────────┼──────────┼──────────┤");
    println!("│ Templated        │ {:8.2} │ {:8.2} │", templated_time, templated_gflops);
    println!("│ Hand-written     │ {:8.2} │ {:8.2} │", handwritten_time, handwritten_gflops);
    println!("└──────────────────┴──────────┴──────────┘");

    println!("\nTemplated is {:.2}x {}", templated_time / handwritten_time,
             if templated_time < handwritten_time { "FASTER" } else { "SLOWER" });
    println!("llama.cpp baseline: ~220 GFLOPS");
}