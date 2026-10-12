use infer_train::ops::gpu::metal::RmsNormMetalContext;

fn main() {
    let ctx = match RmsNormMetalContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to create Metal context: {}", e);
            return;
        }
    };

    // Simple test: 4 elements
    let hidden_dim = 4;
    let eps = 1e-5;

    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let w: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

    println!("Input: {:?}", x);

    // Expected
    let sum: f64 = x.iter().map(|&v| (v * v) as f64).sum();
    let mean = sum / hidden_dim as f64;
    let scale = (1.0 / (mean + eps as f64).sqrt()) as f32;
    println!("Sum of squares: {}", sum);
    println!("Mean: {}", mean);
    println!("Scale: {}", scale);

    let expected: Vec<f32> = x.iter().map(|&v| v * scale).collect();
    println!("Expected: {:?}", expected);

    // Metal
    let result = ctx.rms_norm_f32(&x, &w, hidden_dim, eps).unwrap();
    println!("Metal: {:?}", result);
}
