// Test RMS norm against expected values
fn rms_norm_reference(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    
    let scale = 1.0 / rms;
    x.iter().zip(weight.iter())
        .map(|(&xi, &wi)| xi * scale * wi)
        .collect()
}

fn main() {
    // Test case from actual L2 FFN-normed computation
    // L2-FFN-hidden_before_norm: min=-1.8175 max=1.8345
    // L2-ffn_norm weight: max=6.2188
    
    let input = vec![1.0f32, -1.8175, 1.8345, 0.0, 0.5, -0.5, 1.0, -1.0];
    let weight = vec![1.0f32, 6.2188, 1.0, 1.0, 6.2188, 1.0, 1.0, 1.0];
    
    let output = rms_norm_reference(&input, &weight, 1e-5);
    
    println!("Input: {:?}", input);
    println!("Weight: {:?}", weight);
    println!("Output: {:?}", output);
    
    // Compute RMS
    let sum_sq: f32 = input.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / input.len() as f32 + 1e-5).sqrt();
    println!("RMS: {:.4}", rms);
    println!("Scale: {:.4}", 1.0 / rms);
    
    // Expected outputs:
    // For weight=6.2188, the output should be input * (1/rms) * 6.2188
    let scale = 1.0 / rms;
    for i in 0..input.len() {
        let expected = input[i] * scale * weight[i];
        println!("  output[{}] = {:.4} (expected {:.4})", i, output[i], expected);
    }
    
    // Check if the actual forward pass output matches
    // L2-FFN-normed: min=-17.1135 max=2.5162
    println!("\nExpected output range: [{:.4}, {:.4}]", output.iter().cloned().fold(f32::INFINITY, f32::min), output.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    
    // With input around -1.8 to 1.8 and weight up to 6.2, the output should be around:
    // -1.8 * 6.2 = -11.2 (if rms is around 1.0)
    // But the actual output is -17.1!
    
    // Let me check what RMS value would produce -17.1 with input=-1.8 and weight=6.2:
    // -17.1 = -1.8 * (1/rms) * 6.2
    // => 1/rms = -17.1 / (-1.8 * 6.2) = 1.53
    // => rms = 0.65
    println!("\nIf output=-17.1, input=-1.8, weight=6.2:");
    println!("  1/rms = {:.4}", -17.1 / (-1.8 * 6.2));
    println!("  rms = {:.4}", 1.0 / (-17.1 / (-1.8 * 6.2)));
}