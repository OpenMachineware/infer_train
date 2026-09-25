// Test the fixed templated GEMM kernel
use infer_train::quant::types::BlockQ4K;
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

fn main() {
    println!("=== Testing Fixed Templated GEMM Kernel ===\n");

    let ctx = MetalContext::new().expect("Failed to create MetalContext");

    // Test with small matrix first
    let m = 128;
    let k = 256;
    let n = 64;

    let weights = generate_q4_k_weights(m, k);
    let input: Vec<f32> = (0..(k * n)).map(|i| (i % 10) as f32 * 0.1).collect();

    println!("Testing small matrix (M={}, K={}, N={})", m, k, n);

    match ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input) {
        Ok(result) => {
            println!("✓ Kernel executed successfully");
            println!("  Result size: {} values", result.len());
            println!("  Sample values: {:.4}, {:.4}, {:.4}, {:.4}",
                     result[0], result[100], result[1000], result[5000]);

            // Check for NaN or infinity
            let has_nan = result.iter().any(|&x| x.is_nan());
            let has_inf = result.iter().any(|&x| x.is_infinite());
            let max_val = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));

            println!("  Max absolute value: {}", max_val);
            println!("  Has NaN: {}", has_nan);
            println!("  Has Inf: {}", has_inf);

            if !has_nan && !has_inf && max_val < 1e10 {
                println!("\n✓ Results look reasonable");
            } else {
                println!("\n✗ Results contain problematic values");
            }
        }
        Err(e) => {
            println!("✗ Kernel failed: {}", e);
        }
    }

    // Test larger matrix
    println!("\n=== Testing Larger Matrix ===");
    let m = 4096;
    let k = 4096;
    let n = 128;

    let weights = generate_q4_k_weights(m, k);
    let input: Vec<f32> = (0..(k * n)).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();

    println!("Testing larger matrix (M={}, K={}, N={})", m, k, n);

    match ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input) {
        Ok(result) => {
            println!("✓ Kernel executed successfully");
            println!("  Result size: {} values", result.len());

            let start = std::time::Instant::now();
            let _ = ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input).unwrap();
            let elapsed = start.elapsed();

            let gflops = (2.0 * m as f64 * k as f64 * n as f64) / elapsed.as_secs_f64() / 1e9;
            println!("  Time: {:?}", elapsed);
            println!("  GFLOPS: {:.2}", gflops);

            let max_val = result.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            println!("  Max absolute value: {}", max_val);
        }
        Err(e) => {
            println!("✗ Kernel failed: {}", e);
        }
    }
}