// GPU GEMM llama.cpp Comparison Test
// Compare our Metal kernels with llama.cpp's implementation

use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use infer_train::quant::types::*;
use half::f16;
use std::time::Instant;

fn generate_f32_input(n: usize, k: usize) -> Vec<f32> {
    (0..n * k).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect()
}

fn generate_q4_k_weights(m: usize, k: usize) -> Vec<BlockQ4K> {
    let nb = k / 256;
    (0..m * nb).map(|i| {
        let mut qs = [0u8; 128];
        let mut scales = [0u8; 12];
        for j in 0..128 { qs[j] = ((i * 17 + j * 13) % 256) as u8; }
        for j in 0..12 { scales[j] = ((i * 23 + j * 7) % 256) as u8; }
        BlockQ4K {
            d: f16::from_f32(0.5 + (i % 10) as f32 * 0.1).to_bits(),
            dmin: f16::from_f32(0.1).to_bits(),
            scales,
            qs,
        }
    }).collect()
}

fn main() {
    println!("GPU GEMM Performance vs llama.cpp");
    println!("==================================");
    println!();
    println!("llama.cpp uses: Tiled GEMM with dequantization to FP16");
    println!("Our kernel: Direct K-quant processing without dequantization");
    println!();
    
    let metal_ctx = MetalContext::new().expect("Failed to create Metal context");
    let m = 4096;
    
    println!("Testing Q4_K GEMM across K dimensions (M={}, N=16)", m);
    println!();
    println!("K\tGFLOPS\tNotes");
    
    for k in [256, 512, 1024, 2048, 4096].iter() {
        if k % 256 != 0 { continue; }
        
        let nb = k / 256;
        let weights = generate_q4_k_weights(m, *k);
        let input = generate_f32_input(16, *k);
        
        // Run multiple iterations for stable measurement
        let start = Instant::now();
        let iterations = 10;
        for _ in 0..iterations {
            let _ = metal_ctx.gemm_q4_k_f32(m, 16, *k, &weights, &input);
        }
        let elapsed = start.elapsed().as_micros() as f64 / iterations as f64;
        let flops = 2.0 * m as f64 * 16.0 * *k as f64;
        let gflops = flops / elapsed / 1000.0;
        
        let path = if nb <= 4 { "Small K (per-thread output)" } else { "Large K (block parallel)" };
        
        println!("{}\t{:.0}\t{}", k, gflops, path);
    }
    
    println!();
    println!("To compare with llama.cpp:");
    println!("  1. llama-bench -m <model.q4_k.gguf> -p 16 -n 128");
    println!("  2. Compare prompt processing speed (tokens/sec)");
    println!("  3. Prefill uses GEMM, decode uses MV");
}