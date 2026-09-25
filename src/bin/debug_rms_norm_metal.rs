use infer_train::ops::gpu::metal::RmsNormMetalContext;

fn main() {
    let ctx = RmsNormMetalContext::new().expect("Failed to create Metal context");

    // Test with simple known values
    let hidden_dim = 4;
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let w = vec![1.0f32; hidden_dim];  // Identity weights
    let eps = 1e-5f32;

    println!("Input: {:?}", x);
    println!("Weights: {:?}", w);
    println!("Hidden dim: {}", hidden_dim);
    println!("Eps: {}", eps);

    // Compute expected result
    let sum_sq: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let mean = sum_sq / hidden_dim as f64;
    let scale = 1.0 / (mean + eps as f64).sqrt();
    let expected: Vec<f32> = x.iter().map(|&v| (v as f64 * scale) as f32).collect();

    println!("\nExpected computation:");
    println!("  Sum of squares: {}", sum_sq);
    println!("  Mean: {}", mean);
    println!("  Scale: {}", scale);
    println!("  Expected output: {:?}", expected);

    // Run Metal
    let result = ctx.rms_norm_f32(&x, &w, hidden_dim, eps).expect("Metal failed");
    println!("\nMetal output: {:?}", result);

    // Compute differences
    let max_diff = expected.iter()
        .zip(result.iter())
        .map(|(e, r)| (e - r).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("Max diff: {}", max_diff);

    // Try with larger input
    let hidden_dim2 = 32;
    let x2: Vec<f32> = (0..hidden_dim2).map(|i| (i + 1) as f32).collect();
    let w2 = vec![1.0f32; hidden_dim2];

    println!("\n=== Test with 32 elements ===");
    println!("Input: {:?}", &x2[..8]);

    let sum_sq2: f64 = x2.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let mean2 = sum_sq2 / hidden_dim2 as f64;
    let scale2 = 1.0 / (mean2 + eps as f64).sqrt();
    let expected2: Vec<f32> = x2.iter().map(|&v| (v as f64 * scale2) as f32).collect();

    let result2 = ctx.rms_norm_f32(&x2, &w2, hidden_dim2, eps).expect("Metal failed");

    println!("Expected first 8: {:?}", &expected2[..8]);
    println!("Metal first 8: {:?}", &result2[..8]);

    let max_diff2 = expected2.iter()
        .zip(result2.iter())
        .map(|(e, r)| (e - r).abs())
        .fold(0.0f32, |a, b| a.max(b));

    println!("Max diff: {}", max_diff2);
}
