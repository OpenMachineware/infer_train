// Simple test to debug templated GEMM kernel
use infer_train::quant::types::BlockQ4K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;
use bytemuck::Zeroable;

fn main() {
    println!("Simple templated GEMM debug test");

    let ctx = MetalContext::new().expect("Failed to create Metal context");

    // Very simple test: M=64, N=32, K=256 (one block per row)
    let m = 64;
    let k = 256;  // One Q4_K block
    let n = 32;

    println!("\nTest: M={}, K={}, N={}", m, k, n);

    // Create simple weights: all values = 0
    let mut weights = vec![BlockQ4K::zeroed(); m];

    // Set first block with known values
    weights[0].d = half::f16::from_f32(1.0).to_bits();
    weights[0].dmin = half::f16::from_f32(0.0).to_bits();

    // Set scales to simple values
    // The get_scale_min_k4_just2 function extracts scales from the 12-byte array
    // For simplicity, use scales that give scale = 1
    // sc[0] = q[0] & 63, so we want q[0] = 1
    weights[0].scales[0] = 1;  // scale for first group = 1
    weights[0].scales[1] = 0;
    weights[0].scales[2] = 0;
    weights[0].scales[3] = 0;
    weights[0].scales[4] = 0;  // min for first group = 0
    weights[0].scales[5] = 0;
    weights[0].scales[6] = 0;
    weights[0].scales[7] = 0;

    // Set quantized values to 8 (no bias after dequantization)
    // Actually Q4_K formula: dl * (q & mask), no -8 bias!
    // So q=0 gives value 0, q=15 gives value 15
    // Let's use q=1 for all
    for i in 0..128 {
        weights[0].qs[i] = 0x11; // Both nibbles = 1
    }

    // Create simple input: all 1.0
    let input: Vec<f32> = vec![1.0; k * n];

    println!("Weights: d=1.0, dmin=0, scale[0]=1, qs=1");
    println!("Input: all 1.0");
    println!("Expected output: d * scale * q * sum(input) = 1 * 1 * 1 * K = 256");

    // Run both kernels
    let result_template = ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input).expect("Templated GEMM failed");
    let result_handwritten = ctx.gemm_q4_k_f32(m, n, k, &weights, &input).expect("Hand-written GEMM failed");

    println!("\nResults (first row):");
    for j in 0..n.min(5) {
        println!("  [0,{}]: template={:.2}, handwritten={:.2}, expected=256",
            j, result_template[j], result_handwritten[j]);
    }

    // Check if values are reasonable
    let template_sum: f64 = result_template[0..n].iter().map(|x| *x as f64).sum();
    let handwritten_sum: f64 = result_handwritten[0..n].iter().map(|x| *x as f64).sum();

    println!("\nSum of first row: template={:.2}, handwritten={:.2}", template_sum, handwritten_sum);

    // Test with different q values
    println!("\n--- Test with q=15 (max value) ---");
    for i in 0..128 {
        weights[0].qs[i] = 0xFF; // Both nibbles = 15
    }

    let result_template = ctx.gemm_template_q4_k_f32(m, n, k, &weights, &input).expect("Templated GEMM failed");
    let result_handwritten = ctx.gemm_q4_k_f32(m, n, k, &weights, &input).expect("Hand-written GEMM failed");

    println!("Expected: 1 * 1 * 15 * 256 = 3840");
    println!("Results (first row, first 5 cols):");
    for j in 0..n.min(5) {
        println!("  [0,{}]: template={:.2}, handwritten={:.2}",
            j, result_template[j], result_handwritten[j]);
    }
}